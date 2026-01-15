'''Mediapipe control arm and gripper'''
import time
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from collections import deque
from Robotic_Arm.rm_robot_interface import *

JOINT_QUEUE_LEN = 5    #平衡抖动
joint_queue = deque(maxlen=JOINT_QUEUE_LEN)

# 归一化
def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

# 转四元数
def rotation_matrix_to_quaternion(R: np.ndarray):
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return [qw, qx, qy, qz]

# 1. 初始化 RealSense
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)

# 相机内参
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

# 2. 初始化机械臂 & 算法
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
arm.rm_create_robot_arm("192.168.1.18", 8080)
arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E
force_type = rm_force_type_e.RM_MODEL_RM_B_E
algo = Algo(arm_model=arm_model, force_type=force_type)

# 3. 运动到初始末端位置 
arm.rm_movej([2.3, 12.26, 85.40, 1.52, 81.97, -86.50], 20, 0, 0, 1)
time.sleep(0.2)
# 读取初始位置
pose0 = arm.rm_get_current_arm_state()[1]['pose'][:3]      # 只取x y z

# 手–眼标定旋转
q = np.array([-0.489673, 0.51976, -0.485529, 0.504314])
w, x, y, z = q
R_rc = np.array([
    [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
    [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
    [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
], dtype=np.float32)

# 4. MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9,
                       min_tracking_confidence=0.9,
                       max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 5. 零位标定
Pc0 = None
scale = 0.8  # 映射灵敏度，可调
start_time = time.time()
print("请将手腕移到“零位”并保持不动, 3秒后开始标定…")
while Pc0 is None:
    if time.time() - start_time < 3.0:
        time.sleep(0.1)
        continue
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    df, cf = aligned.get_depth_frame(), aligned.get_color_frame()
    if not df or not cf:
        continue

    color = np.asanyarray(cf.get_data())
    rgb   = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)
    if not res.multi_hand_landmarks:
        cv2.imshow("Init Zero Pose (Q退出)", color)
        cv2.waitKey(1)
        continue

    lm = res.multi_hand_landmarks[0]
    u = int(lm.landmark[mp_hands.HandLandmark.WRIST].x * intr.width)
    v = int(lm.landmark[mp_hands.HandLandmark.WRIST].y * intr.height)
    u, v = np.clip(u,0,intr.width-1), np.clip(v,0,intr.height-1)
    D = df.get_distance(u, v)
    Pc0 = np.array([(u-cx)*D/fx, (v-cy)*D/fy, D], dtype=np.float32)
    print("标定零位 Pc0 =", Pc0)
    cv2.destroyAllWindows()

# 6. 主循环：增量映射 + 姿态 + 逆解 + rm_movej_follow
cv2.namedWindow("Hand→Arm Joint PBVS", cv2.WINDOW_AUTOSIZE)
try:
    while True:
        frames  = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df, cf = aligned.get_depth_frame(), aligned.get_color_frame()
        if not df or not cf:
            continue

        color   = np.asanyarray(cf.get_data())
        display = color.copy()
        rgb     = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        res     = hands.process(rgb)
        h, w, _ = color.shape
        

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(display, lm, mp_hands.HAND_CONNECTIONS)

            # #---------------控制夹抓-------------------------------  
            # # 拇指指尖与食指指尖
            # th = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
            # idx= lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # u_t, v_t = int(th.x*w), int(th.y*h)
            # u_i, v_i = int(idx.x*w), int(idx.y*h)
            # # 把坐标 clamp 到 [0, w-1], [0, h-1]
            # u_t = np.clip(u_t, 0, w-1)
            # v_t = np.clip(v_t, 0, h-1)
            # u_i = np.clip(u_i, 0, w-1)
            # v_i = np.clip(v_i, 0, h-1)
            # # 3. 投影到三维
            # z_t = df.get_distance(u_t, v_t)
            # z_i = df.get_distance(u_i, v_i)
            # x_t = (u_t - cx) * z_t / fx
            # y_t = (v_t - cy) * z_t / fy
            # x_i = (u_i - cx) * z_i / fx
            # y_i = (v_i - cy) * z_i / fy
            # # 4. 计算距离 & 滤波
            # d = np.linalg.norm([x_t-x_i, y_t-y_i, z_t-z_i])
            # # 当 MediaPipe 检测距离小于 4.5cm 时，强制当作闭合0,反之则张开
            # if d < 0.045: 
            #     arm.rm_set_gripper_position(0, block=True, timeout=0)
            # else:
            #     arm.rm_set_gripper_position(1000, block=True, timeout=0)
            # #------------------------------------------------------------


            # 6.1 反投影mediapipe的关键点到相机坐标系下的真实 3D 坐标
            def unproject(lm):
                u = int(lm.x * intr.width)
                v = int(lm.y * intr.height)
                u, v = np.clip(u,0,intr.width-1), np.clip(v,0,intr.height-1)
                D = df.get_distance(u, v)    # 深度（米）
                X = (u - cx) * D / fx
                Y = (v - cy) * D / fy
                Z = D
                return np.array([X, Y, Z], dtype=np.float32)
            
            # 手腕、中指根（MIDDLE_FINGER_MCP）、拇指根（THUMB_MCP）
            Wc  = unproject(lm.landmark[mp_hands.HandLandmark.WRIST])
            Fc  = unproject(lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
            Tc  = unproject(lm.landmark[mp_hands.HandLandmark.THUMB_MCP]) 

            # 6.2 增量映射 ΔPr
            deltaPr = R_rc.dot(Wc - Pc0) * scale

            # 6.3 姿态计算(用真实 3D 坐标构建手部局部坐标系) 
            z_ax = normalize(Fc - Wc)          # Z轴：腕→中指根(因为mediapipe里手腕点到中指根点是近似垂直的)
            tmp  = Tc - Wc                     # 腕→拇指根
            x_ax = normalize(tmp - np.dot(tmp, z_ax)*z_ax)   # 去掉在 z_ax 上的分量
            y_ax = np.cross(z_ax, x_ax)

            R_hand = np.stack([x_ax, y_ax, z_ax], axis=1)
            R_base = R_rc.dot(R_hand)

            quat   = rotation_matrix_to_quaternion(R_base)
            # ---- 四元数归一化 ----
            qv = np.array(quat, dtype=np.float64)
            qv /= np.linalg.norm(qv)
            quat = qv.tolist()
            eul    = algo.rm_algo_quaternion2euler(quat)  # [rx, ry, rz]

            # 6.4 目标末端位姿
            target_pos  = pose0 + deltaPr
            target_pose = np.concatenate([target_pos, eul]).tolist()
            q_current = arm.rm_get_joint_degree()[1]
            # print(f"目标末端位姿:{target_pose}")
            # 6.5 构造逆解参数并调用逆解
            params = rm_inverse_kinematics_params_t(q_current, target_pose,1)
            ret_ik, joints = algo.rm_algo_inverse_kinematics(params)
            # print(f"code:{ret_ik},关节角度:{joints}")
            if ret_ik == 0:
                # 推入队列
                joint_queue.append(joints)
                # 计算每个关节的平均值
                avg_joints = np.mean(joint_queue, axis=0).tolist()
                # 6.6 平滑后关节空间跟随下发
                arm.rm_movej_follow(avg_joints)
            else:
                print(f"code:{ret_ik},逆解失败")

        cv2.imshow("Hand→Arm Joint PBVS", display)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    hands.close()
    arm.rm_delete_robot_arm()
    cv2.destroyAllWindows()   