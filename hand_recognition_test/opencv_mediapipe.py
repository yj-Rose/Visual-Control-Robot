import cv2
import mediapipe as mp

# 初始化 MediaPipe 手部检测模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 配置参数：最多检测1只手，模型检测置信度和追踪置信度都为0.7
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 打开摄像头（默认设备为 0）
cap = cv2.VideoCapture(4)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 翻转图像（因为摄像头是镜像的）
    frame = cv2.flip(frame, 1)
    
    # 转换为 RGB（MediaPipe 要求）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理帧
    results = hands.process(rgb_frame)

    # 如果检测到手
    if results.multi_hand_landmarks and results.multi_handedness:
    	 # 遍历每只手的关键点和左右手信息
    	 for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
        #for hand_landmarks in results.multi_hand_landmarks:
            # 绘制关键点和连接线
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 获取左右手信息
            hand_label = handedness.classification[0].label  # "Left" 或 "Right"
            # 根据左右手设置不同颜色（左手蓝色，右手红色）
            color = (255, 0, 0) if hand_label == "Left" else (0, 0, 255)

            # 在手腕位置（关键点0）附近显示左右手标签
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, hand_label, (wrist_x - 20, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 打印部分关键点坐标（可用于后续控制）
            for i, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                if i == 8:  # 例如食指指尖（id=8）
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"Index Tip: {x},{y}", (x+10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示图像
    cv2.imshow('Hand Tracking', frame)

    # 按下 ESC 或 q 退出
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
