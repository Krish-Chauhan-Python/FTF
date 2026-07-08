import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import get_model

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


YOLO_CONF_THRESHOLD = 0.65
TARGET_CLASSES = {"test tube", "test_tube", "test-tube", "beaker"}
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
ARM_JOINTS = [name for name in MOTORS if name != "gripper"]


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


NORMALIZED_TARGET_CLASSES = {normalize_label(name) for name in TARGET_CLASSES}


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))





class LiveTorqueEstimator:
    def __init__(
        self,
        camera_index: int,
        yolo_model_path: str,
        torque_model_path: str,
        backbone: str,
        yolo_imgsz: int,
        yolo_conf: float,
        smoothing_window: int,
        show_camera: bool,
    ) -> None:
        self.device = torch.device("cpu")
        self.yolo = YOLO(yolo_model_path)
        self.class_names = self._get_class_name_map(self.yolo.names)
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        self.show_camera = show_camera

        self.regressor = get_model("yolo_pipeline", backbone=backbone, pretrained=False)
        state_dict = torch.load(torque_model_path, map_location=self.device)
        self.regressor.load_state_dict(state_dict)
        self.regressor.to(self.device)
        self.regressor.eval()

        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self.predictions = deque(maxlen=max(1, smoothing_window))

    @staticmethod
    def _get_class_name_map(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v).strip().lower() for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(name).strip().lower() for i, name in enumerate(names)}
        return {}

    @staticmethod
    def _preprocess_crop(crop_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
        crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)

        tensor = tensor.to(device)
        tensor = (tensor - mean) / std
        return tensor

    def _find_best_detection(self, result) -> Optional[Tuple[Tuple[int, int, int, int], float, str]]:
        if result.boxes is None or len(result.boxes) == 0:
            return None

        best = None
        best_conf = -1.0

        for box in result.boxes:
            conf = float(box.conf.item())
            cls_idx = int(box.cls.item())
            cls_name = self.class_names.get(cls_idx, "")

            if conf < self.yolo_conf:
                continue
            if normalize_label(cls_name) not in NORMALIZED_TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if conf > best_conf:
                best_conf = conf
                best = (bbox, conf, cls_name)

        return best

    def predict_torque(self) -> Optional[float]:
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return None

        with torch.no_grad():
            result = self.yolo.predict(
                source=frame_bgr,
                conf=self.yolo_conf,
                imgsz=self.yolo_imgsz,
                device="cpu",
                verbose=False,
            )[0]

            detection = self._find_best_detection(result)
            if detection is None:
                self.predictions.clear()
                if self.show_camera:
                    cv2.putText(frame_bgr, "No beaker/test-tube detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Live Torque Estimator", frame_bgr)
                    cv2.waitKey(1)
                return None

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
            model_input = self._preprocess_crop(crop_rgb, self.device)
            torque_pred = float(self.regressor(model_input).item())
            self.predictions.append(torque_pred)

            smooth_torque = float(np.mean(self.predictions))
            if cls_name in ("test tube", "test_tube", "test-tube"):
                smooth_torque = smooth_torque * 10 + 20
            elif cls_name == "beaker":
                smooth_torque = smooth_torque * 10 + 30
            smooth_torque += 5

            if self.show_camera:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Class: {cls_name} ({conf:.2f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Pred Torque: {smooth_torque:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Live Torque Estimator", frame_bgr)
                cv2.waitKey(1)

            return smooth_torque

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        if self.show_camera:
            cv2.destroyAllWindows()


def load_snapshots(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data)!r}")
    return data


def get_positions_from_snapshot(snapshot: dict[str, object]) -> dict[str, float]:
    positions = snapshot.get("positions")
    if not isinstance(positions, dict):
        raise TypeError(f"Invalid snapshot positions payload: {type(positions)!r}")

    result: dict[str, float] = {}
    for motor_name in MOTORS:
        if motor_name in positions:
            result[motor_name] = float(positions[motor_name])
    return result


def read_gripper_load_abs(bus: FeetechMotorsBus) -> int:
    raw_load = bus.read("Present_Load", "gripper", normalize=False)
    return abs(int(raw_load))


def write_full_goal(bus: FeetechMotorsBus, arm_positions: dict[str, float], gripper_goal: float) -> None:
    goals: dict[str, float] = {joint: arm_positions[joint] for joint in ARM_JOINTS if joint in arm_positions}
    goals["gripper"] = gripper_goal
    bus.sync_write("Goal_Position", goals, normalize=False)


def map_torque_to_limits(
    predicted_torque: float,
    torque_min: float,
    torque_max: float,
    limit_min: int,
    limit_max: int,
    current_min: int,
    current_max: int,
) -> tuple[int, int]:
    span = max(1e-6, torque_max - torque_min)
    ratio = clamp((predicted_torque - torque_min) / span, 0.0, 1.0)
    max_torque_limit = int(round(limit_min + ratio * (limit_max - limit_min)))
    protection_current = int(round(current_min + ratio * (current_max - current_min)))
    return max_torque_limit, protection_current


def apply_dynamic_gripper_limits(
    bus: FeetechMotorsBus,
    estimator: LiveTorqueEstimator,
    torque_min: float,
    torque_max: float,
    limit_min: int,
    limit_max: int,
    current_min: int,
    current_max: int,
    cache: dict[str, int],
) -> None:
    pred = estimator.predict_torque()
    if pred is None:
        return

    max_torque_limit, protection_current = map_torque_to_limits(
        pred,
        torque_min,
        torque_max,
        limit_min,
        limit_max,
        current_min,
        current_max,
    )

    if cache.get("max_torque_limit") != max_torque_limit:
        bus.write("Max_Torque_Limit", "gripper", max_torque_limit, normalize=False)
        cache["max_torque_limit"] = max_torque_limit

    if cache.get("protection_current") != protection_current:
        bus.write("Protection_Current", "gripper", protection_current, normalize=False)
        cache["protection_current"] = protection_current

    print(
        f"\r[MODEL] pred_torque={pred:.3f} -> Max_Torque_Limit={max_torque_limit}, Protection_Current={protection_current}",
        end="",
        flush=True,
    )


def move_to_pose_fixed_gripper(
    bus: FeetechMotorsBus,
    target_arm_pose: dict[str, float],
    gripper_goal: float,
    transition_steps: int,
    transition_step_delay: float,
    estimator: LiveTorqueEstimator,
    dyn_cfg: dict[str, float | int],
    cache: dict[str, int],
) -> None:
    current_raw = bus.sync_read("Present_Position", normalize=False)
    start_arm_pose = {joint: float(current_raw[joint]) for joint in ARM_JOINTS}

    for step in range(1, max(1, transition_steps) + 1):
        t = smoothstep(step / max(1, transition_steps))
        interp: dict[str, float] = {}
        for joint in ARM_JOINTS:
            start_val = start_arm_pose[joint]
            end_val = target_arm_pose.get(joint, start_val)
            interp[joint] = start_val + (end_val - start_val) * t

        apply_dynamic_gripper_limits(
            bus,
            estimator,
            float(dyn_cfg["torque_min"]),
            float(dyn_cfg["torque_max"]),
            int(dyn_cfg["limit_min"]),
            int(dyn_cfg["limit_max"]),
            int(dyn_cfg["current_min"]),
            int(dyn_cfg["current_max"]),
            cache,
        )
        write_full_goal(bus, interp, gripper_goal)
        time.sleep(max(0.0, transition_step_delay))

    print()


def close_gripper_until_load(
    bus: FeetechMotorsBus,
    initial_gripper_goal: float,
    target_load: int,
    close_step: float,
    min_gripper_goal: float,
    poll_delay: float,
    estimator: LiveTorqueEstimator,
    dyn_cfg: dict[str, float | int],
    cache: dict[str, int],
) -> tuple[float, int]:
    gripper_goal = float(initial_gripper_goal)
    bus.write("Goal_Position", "gripper", gripper_goal, normalize=False)

    load_abs = read_gripper_load_abs(bus)
    while load_abs < target_load and gripper_goal > min_gripper_goal:
        apply_dynamic_gripper_limits(
            bus,
            estimator,
            float(dyn_cfg["torque_min"]),
            float(dyn_cfg["torque_max"]),
            int(dyn_cfg["limit_min"]),
            int(dyn_cfg["limit_max"]),
            int(dyn_cfg["current_min"]),
            int(dyn_cfg["current_max"]),
            cache,
        )

        gripper_goal = max(min_gripper_goal, gripper_goal - close_step)
        bus.write("Goal_Position", "gripper", gripper_goal, normalize=False)
        time.sleep(max(0.0, poll_delay))
        load_abs = read_gripper_load_abs(bus)
        print(
            f"\r[LOAD SEEK] target={target_load} current={load_abs} gripper_goal={gripper_goal:.1f}",
            end="",
            flush=True,
        )

    print()
    return gripper_goal, load_abs


def transition_with_constant_load_control(
    bus: FeetechMotorsBus,
    start_pose: dict[str, float],
    end_pose: dict[str, float],
    gripper_goal: float,
    target_load: int,
    load_deadband: int,
    gripper_adjust_step: float,
    min_gripper_goal: float,
    max_gripper_goal: float,
    transition_steps: int,
    transition_step_delay: float,
    estimator: LiveTorqueEstimator,
    dyn_cfg: dict[str, float | int],
    cache: dict[str, int],
) -> float:
    steps = max(1, transition_steps)
    for step in range(1, steps + 1):
        t = smoothstep(step / steps)
        arm_interp: dict[str, float] = {}
        for joint in ARM_JOINTS:
            start_val = start_pose.get(joint, 0.0)
            end_val = end_pose.get(joint, start_val)
            arm_interp[joint] = start_val + (end_val - start_val) * t

        apply_dynamic_gripper_limits(
            bus,
            estimator,
            float(dyn_cfg["torque_min"]),
            float(dyn_cfg["torque_max"]),
            int(dyn_cfg["limit_min"]),
            int(dyn_cfg["limit_max"]),
            int(dyn_cfg["current_min"]),
            int(dyn_cfg["current_max"]),
            cache,
        )

        load_abs = read_gripper_load_abs(bus)
        if load_abs < target_load - load_deadband:
            gripper_goal -= gripper_adjust_step
        elif load_abs > target_load + load_deadband:
            gripper_goal += gripper_adjust_step

        gripper_goal = clamp(gripper_goal, min_gripper_goal, max_gripper_goal)
        write_full_goal(bus, arm_interp, gripper_goal)

        print(
            f"\r[TRANSITION] step={step}/{steps} load={load_abs} target={target_load} gripper_goal={gripper_goal:.1f}",
            end="",
            flush=True,
        )
        time.sleep(max(0.0, transition_step_delay))

    print()
    return gripper_goal


def ramp_gripper_to_goal(
    bus: FeetechMotorsBus,
    start_goal: float,
    end_goal: float,
    ramp_steps: int,
    ramp_step_delay: float,
    estimator: LiveTorqueEstimator,
    dyn_cfg: dict[str, float | int],
    cache: dict[str, int],
) -> float:
    steps = max(1, int(ramp_steps))
    delay = max(0.0, float(ramp_step_delay))

    for step in range(1, steps + 1):
        t = smoothstep(step / steps)
        gripper_goal = start_goal + (end_goal - start_goal) * t

        apply_dynamic_gripper_limits(
            bus,
            estimator,
            float(dyn_cfg["torque_min"]),
            float(dyn_cfg["torque_max"]),
            int(dyn_cfg["limit_min"]),
            int(dyn_cfg["limit_max"]),
            int(dyn_cfg["current_min"]),
            int(dyn_cfg["current_max"]),
            cache,
        )

        bus.write("Goal_Position", "gripper", gripper_goal, normalize=False)
        print(
            f"\r[FINAL RAMP] step={step}/{steps} gripper_goal={gripper_goal:.1f}",
            end="",
            flush=True,
        )
        time.sleep(delay)

    print()
    return end_goal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merged flow: live camera torque prediction + 5-pose gripper load sequence"
    )

    parser.add_argument("--port", type=str, default="COM5", help="Serial port for SO101 controller")
    parser.add_argument("--readings-file", type=str, default="servo_readings.json", help="Path to saved snapshots JSON")

    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--yolo-model", type=str, default="testing/best.pt", help="Path to YOLO weights")
    parser.add_argument("--torque-model", type=str, default="best_resnet_torque.pth", help="Path to torque model")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONF_THRESHOLD, help="YOLO confidence threshold")
    parser.add_argument("--smoothing-window", type=int, default=8, help="Moving average window for predicted torque")
    parser.add_argument("--show-camera", action="store_true", help="Show live camera preview window")

    parser.add_argument("--pred-torque-min", type=float, default=0.0, help="Lower bound for predicted torque mapping")
    parser.add_argument("--pred-torque-max", type=float, default=40.0, help="Upper bound for predicted torque mapping")
    parser.add_argument("--torque-limit-min", type=int, default=300, help="Mapped min Max_Torque_Limit")
    parser.add_argument("--torque-limit-max", type=int, default=950, help="Mapped max Max_Torque_Limit")
    parser.add_argument("--protection-current-min", type=int, default=150, help="Mapped min Protection_Current")
    parser.add_argument("--protection-current-max", type=int, default=500, help="Mapped max Protection_Current")

    parser.add_argument("--target-load", type=int, default=200, help="Absolute Present_Load setpoint to maintain")
    parser.add_argument("--load-deadband", type=int, default=5, help="Allowed load error band before adjusting gripper")
    parser.add_argument("--initial-gripper-goal", type=float, default=3000.0, help="Initial gripper goal position")
    parser.add_argument("--final-gripper-goal", type=float, default=3000.0, help="Final gripper goal at end of sequence")
    parser.add_argument("--close-step", type=float, default=10.0, help="Amount to decrease gripper per load-seek step")
    parser.add_argument("--gripper-adjust-step", type=float, default=4.0, help="Adjustment step while maintaining load")
    parser.add_argument("--min-gripper-goal", type=float, default=0.0, help="Minimum gripper goal clamp")
    parser.add_argument("--max-gripper-goal", type=float, default=4095.0, help="Maximum gripper goal clamp")
    parser.add_argument("--transition-steps", type=int, default=60, help="Interpolation steps between poses")
    parser.add_argument("--transition-step-delay", type=float, default=0.02, help="Delay per interpolation step")
    parser.add_argument("--poll-delay", type=float, default=0.03, help="Delay during gripper load seek")
    parser.add_argument("--final-ramp-steps", type=int, default=80, help="Interpolation steps for final gripper ramp")
    parser.add_argument("--final-ramp-step-delay", type=float, default=0.02, help="Delay per step during final gripper ramp")

    args = parser.parse_args()

    snapshots = load_snapshots(Path(args.readings_file))
    if len(snapshots) < 5:
        raise ValueError("Need at least 5 snapshots in the JSON file")

    poses = [get_positions_from_snapshot(snapshots[i]) for i in range(5)]

    dyn_cfg: dict[str, float | int] = {
        "torque_min": args.pred_torque_min,
        "torque_max": args.pred_torque_max,
        "limit_min": args.torque_limit_min,
        "limit_max": args.torque_limit_max,
        "current_min": args.protection_current_min,
        "current_max": args.protection_current_max,
    }

    estimator = LiveTorqueEstimator(
        camera_index=args.camera_index,
        yolo_model_path=args.yolo_model,
        torque_model_path=args.torque_model,
        backbone=args.backbone,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        smoothing_window=args.smoothing_window,
        show_camera=args.show_camera,
    )

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    limit_cache: dict[str, int] = {}

    try:
        bus.connect(handshake=False)
        bus.enable_torque()

        initial_goal = clamp(args.initial_gripper_goal, args.min_gripper_goal, args.max_gripper_goal)
        print("[STEP] Moving to position 1 with gripper set to initial goal")
        move_to_pose_fixed_gripper(
            bus,
            target_arm_pose=poses[0],
            gripper_goal=initial_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
            estimator=estimator,
            dyn_cfg=dyn_cfg,
            cache=limit_cache,
        )

        input("[INPUT] Press Enter to start closing gripper until load setpoint is reached...")
        gripper_goal, reached_load = close_gripper_until_load(
            bus,
            initial_gripper_goal=initial_goal,
            target_load=args.target_load,
            close_step=args.close_step,
            min_gripper_goal=args.min_gripper_goal,
            poll_delay=args.poll_delay,
            estimator=estimator,
            dyn_cfg=dyn_cfg,
            cache=limit_cache,
        )
        print(f"[INFO] Load seek done: load={reached_load}, gripper_goal={gripper_goal:.1f}")

        input("[INPUT] Press Enter to move to position 2...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[0],
            end_pose=poses[1],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
            estimator=estimator,
            dyn_cfg=dyn_cfg,
            cache=limit_cache,
        )

        input("[INPUT] Press Enter to move to position 3...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[1],
            end_pose=poses[2],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
            estimator=estimator,
            dyn_cfg=dyn_cfg,
            cache=limit_cache,
        )

        input("[INPUT] Press Enter to move to position 4...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[2],
            end_pose=poses[3],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
            estimator=estimator,
            dyn_cfg=dyn_cfg,
            cache=limit_cache,
        )

        input("[INPUT] Press Enter to move to position 5...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[3],
            end_pose=poses[4],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
            estimator=estimator,
            dyn_cfg=dyn_cfg,
            cache=limit_cache,
        )

        final_goal = clamp(args.final_gripper_goal, args.min_gripper_goal, args.max_gripper_goal)
        gripper_goal = ramp_gripper_to_goal(
            bus,
            start_goal=gripper_goal,
            end_goal=final_goal,
            ramp_steps=args.final_ramp_steps,
            ramp_step_delay=args.final_ramp_step_delay,
            estimator=estimator,
            dyn_cfg=dyn_cfg,
            cache=limit_cache,
        )
        print(f"[DONE] Position 5 reached. Gripper slowly ramped to {gripper_goal:.1f}")

    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        estimator.close()


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
