import argparse
import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import get_model


YOLO_CONF_THRESHOLD = 0.60
TARGET_CLASSES = {"test tube", "test_tube", "test-tube", "beaker"}
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

CAMERA_BACKEND_MAP = {
    "any": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


NORMALIZED_TARGET_CLASSES = {normalize_label(name) for name in TARGET_CLASSES}





def get_class_name_map(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v).strip().lower() for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(name).strip().lower() for i, name in enumerate(names)}
    return {}


def find_best_target_detection(
    result,
    class_names: Dict[int, str],
    conf_threshold: float,
) -> Optional[Tuple[Tuple[int, int, int, int], float, str]]:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = None
    best_conf = -1.0

    for box in result.boxes:
        conf = float(box.conf.item())
        cls_idx = int(box.cls.item())
        cls_name = class_names.get(cls_idx, "")

        if conf < conf_threshold:
            continue
        if normalize_label(cls_name) not in NORMALIZED_TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = (int(x1), int(y1), int(x2), int(y2))

        if conf > best_conf:
            best_conf = conf
            best = (bbox, conf, cls_name)

    return best


def preprocess_crop(crop_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)

    tensor = tensor.to(device)
    tensor = (tensor - mean) / std
    return tensor


def draw_text(frame: np.ndarray, text: str, org: Tuple[int, int], color=(0, 255, 0)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def open_camera(camera_index: int, backend_name: str) -> cv2.VideoCapture:
    backend = CAMERA_BACKEND_MAP[backend_name]
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Live torque prediction from webcam using YOLO + ResNet regressor")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--yolo-model", type=str, default="testing/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--torque-model", type=str, default="best_resnet_torque.pth", help="Path to trained torque model")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONF_THRESHOLD, help="YOLO confidence threshold")
    parser.add_argument("--smoothing-window", type=int, default=8, help="Moving-average window for predicted torque")
    parser.add_argument(
        "--camera-backend",
        type=str,
        default="msmf",
        choices=["msmf", "dshow", "any"],
        help="Camera backend on Windows. Use dshow if MSMF grabFrame warnings appear.",
    )
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=15,
        help="How many consecutive frame read failures before camera reconnect",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow window (useful for headless environments)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print("[INFO] Using CPU for live inference")

    print(f"[INFO] Loading YOLO model from: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)
    class_names = get_class_name_map(yolo.names)

    target_ids = {
        idx
        for idx, name in class_names.items()
        if normalize_label(name) in NORMALIZED_TARGET_CLASSES
    }
    if not target_ids:
        raise RuntimeError(
            "No Beaker/Test-Tube class found in YOLO model labels. "
            f"Available labels: {class_names}"
        )

    print(f"[INFO] Loading torque regressor from: {args.torque_model}")
    regressor = get_model("yolo_pipeline", backbone=args.backbone, pretrained=False)
    state_dict = torch.load(args.torque_model, map_location=device)
    regressor.load_state_dict(state_dict)
    regressor.to(device)
    regressor.eval()

    cap = open_camera(args.camera_index, args.camera_backend)
    if not cap.isOpened():
        if args.camera_backend == "msmf":
            print("[WARN] MSMF camera open failed. Trying DSHOW backend...")
            cap = open_camera(args.camera_index, "dshow")
            if cap.isOpened():
                print("[INFO] Camera opened with DSHOW backend")
                args.camera_backend = "dshow"
            else:
                raise RuntimeError(f"Could not open camera index {args.camera_index} with MSMF or DSHOW")
        else:
            raise RuntimeError(f"Could not open camera index {args.camera_index}")

    consecutive_read_failures = 0
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    predictions = deque(maxlen=max(1, args.smoothing_window))
    can_display = not args.no_display

    print("[INFO] Live prediction started. Press 'q' to quit.")
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                consecutive_read_failures += 1
                if consecutive_read_failures == 1:
                    print(
                        f"[WARN] Camera read failed on backend '{args.camera_backend}'. "
                        "Will retry and reconnect if needed."
                    )

                if consecutive_read_failures >= max(1, args.max_read_failures):
                    cap.release()
                    time.sleep(0.2)
                    cap = open_camera(args.camera_index, args.camera_backend)

                    if not cap.isOpened() and args.camera_backend == "msmf":
                        print("[WARN] MSMF reconnect failed. Switching to DSHOW...")
                        cap = open_camera(args.camera_index, "dshow")
                        if cap.isOpened():
                            args.camera_backend = "dshow"
                            print("[INFO] Camera reconnected using DSHOW backend")

                    consecutive_read_failures = 0

                if can_display:
                    try:
                        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        draw_text(error_frame, "Camera read failed. Retrying...", (20, 40), color=(0, 0, 255))
                        cv2.imshow("Live Torque Predictor", error_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        can_display = False
                        print(
                            "[WARN] OpenCV display is unavailable. Install GUI-enabled OpenCV with: "
                            "python -m pip install --upgrade opencv-python"
                        )
                continue

            consecutive_read_failures = 0

            result = yolo.predict(
                source=frame_bgr,
                conf=args.yolo_conf,
                imgsz=args.yolo_imgsz,
                device="cpu",
                verbose=False,
            )[0]

            detection = find_best_target_detection(result, class_names, args.yolo_conf)

            if detection is None:
                draw_text(frame_bgr, "No beaker/test-tube detected", (20, 30), color=(0, 0, 255))
                predictions.clear()
            else:
                (x1, y1, x2, y2), conf, cls_name = detection
                h, w = frame_bgr.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(1, min(y2, h))

                if x2 > x1 and y2 > y1:
                    crop_bgr = frame_bgr[y1:y2, x1:x2]
                else:
                    crop_bgr = frame_bgr

                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                model_input = preprocess_crop(crop_rgb, device=device)
                torque_pred = float(regressor(model_input).item())
                predictions.append(torque_pred)

                smooth_torque = float(np.mean(predictions))

                    
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_text(frame_bgr, f"Class: {cls_name} ({conf:.2f})", (20, 30))
                draw_text(frame_bgr, f"Torque: {smooth_torque+5:.4f}", (20, 60), color=(255, 255, 0))

            if can_display:
                try:
                    cv2.imshow("Live Torque Predictor", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    can_display = False
                    print(
                        "[WARN] OpenCV display is unavailable. Install GUI-enabled OpenCV with: "
                        "python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless; "
                        "python -m pip install --upgrade opencv-python"
                    )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Keep OpenCV CPU threading conservative.
    cv2.setNumThreads(1)
    main()
