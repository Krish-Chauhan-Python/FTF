import cv2

path = r"C:\Users\Hi Krish\Downloads\RH20T_cfg1\RH20T_cfg1\task_0030_user_0010_scene_0009_cfg_0001\cam_750612070853\color.mp4"

cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("Failed to open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("color", frame)

    # 1 ms delay, press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
