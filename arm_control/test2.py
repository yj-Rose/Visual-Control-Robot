# 用多个指根（MCP）做位置控制 + 深度中值 + EMA 平滑 + 跳变限幅
import time
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from collections import deque
from Robotic_Arm.rm_robot_interface import *

JOINT_QUEUE_LEN = 5
joint_queue = deque(maxlen=JOINT_QUEUE_LEN)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def rotation_matrix_to_quaternion(R: np.ndarray):
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw + 1e-12)
    qy = (R[0,2] - R[2,0]) / (4*qw + 1e-12)
    qz = (R[1,0] - R[0,1]) / (4*qw + 1e-12)
    return [qw, qx, qy, qz]

# ----- RealSense / Robot / Mediapipe 初始化（沿用你原设置） -----
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)

color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
arm.rm_create_robot_arm("192.168.1.18", 8080)
arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E
force_type = rm_force_type_e.RM_MODEL_RM_B_E
algo = Algo(arm_model=arm_model, force_type=force_type)

arm.rm_movej([2.3, 12.26, 85.40, 1.52, 81.97, -86.50], 20, 0, 0, 1)
time.sleep(0.5)
pose0 = np.array(arm.rm_get_current_arm_state()[1]['pose'][:3], dtype=np.float64)

q = np.array([-0.489673, 0.51976, -0.485529, 0.504314])
w, x, y, z = q
R_rc = np.array([
    [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
    [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
    [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
], dtype=np.float32)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.88, min_tracking_confidence=0.88, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# ----- 新增：参数 -----
# 使用哪些 landmark（MCP 点）： index=5 middle=9 ring=13
POS_LANDMARK_IDS = [0,5,9,13]      # 可调整为 [5],[9],[13] 或包含其他点
DEPTH_MEDIAN_WINDOW = 1            # 半径：1 => 3x3 窗口做中值
EMA_ALPHA_POS = 0.5               # 位置 EMA 强度（0.0..1.0，越小越平滑）
MAX_STEP_PER_FRAME = 0.06          # 每帧位置最大允许移动（米），防抖跳变
SCALE = 1.0

# EMA 状态
pos_ema = None

def depth_median(df, u, v, win=1):
    # 获取 (2*win+1)^2 邻域内的深度值并返回非零中位数，若无有效深度返回 0
    hvals = []
    for dy in range(-win, win+1):
        for dx in range(-win, win+1):
            uu = int(np.clip(u+dx, 0, intr.width-1))
            vv = int(np.clip(v+dy, 0, intr.height-1))
            try:
                d = df.get_distance(uu, vv)
            except:
                d = 0
            if d and (not np.isnan(d)) and d>0.0001:
                hvals.append(d)
    if len(hvals)==0:
        return 0.0
    return float(np.median(hvals))

def unproject_pixel(u, v, d):
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    return np.array([X, Y, Z], dtype=np.float64)

def get_stable_hand_pos_from_MCPs(lm, df, intr, ids=POS_LANDMARK_IDS):
    """
    用多个 MCP 点（indices）计算一个稳健的 3D 点：
      - 对每个 landmark 做深度中值取样 (depth_median)
      - 只用有效 depth 点
      - 返回这些点的均值（或中位数），若没有有效点则返回 None
    """
    pts = []
    for idx in ids:
        l = lm.landmark[idx]
        # 判断 landmark 是否在图像内（Mediapipe 有时给 <0 或 >1）
        if l.x < -0.05 or l.x > 1.05 or l.y < -0.05 or l.y > 1.05:
            continue
        u = int(l.x * intr.width); v = int(l.y * intr.height)
        u = int(np.clip(u, 0, intr.width-1)); v = int(np.clip(v, 0, intr.height-1))
        d_med = depth_median(df, u, v, win=DEPTH_MEDIAN_WINDOW)
        if d_med <= 0.001:   # 无效深度
            continue
        pts.append(unproject_pixel(u, v, d_med))
    if len(pts) == 0:
        return None
    # 使用均值（也可换成 np.median(np.vstack(pts), axis=0)）
    return np.mean(np.vstack(pts), axis=0)

# ----- 零位标定（改成用 MCP 平均） -----
Pc0 = None
start_time = time.time()
print("请将 手 指根（MCP） 或 手腕 置于零位并保持 2 秒，开始零点标定...")
while Pc0 is None:
    if time.time() - start_time < 2.0:
        time.sleep(0.05); continue
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    df, cf = aligned.get_depth_frame(), aligned.get_color_frame()
    if not df or not cf: continue
    color = np.asanyarray(cf.get_data()); rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        cv2.imshow("Init Zero Pose (Q退出)", color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    lm = res.multi_hand_landmarks[0]
    # 优先用 MCP 集合，否则回退到 wrist
    P = get_stable_hand_pos_from_MCPs(lm, df, intr)
    if P is None:
        continue
    Pc0 = P
    print("标定零位 Pc0 =", Pc0)
cv2.destroyAllWindows()

# ----- 主循环 -----
cv2.namedWindow("Hand→Arm Joint PBVS", cv2.WINDOW_AUTOSIZE)
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df, cf = aligned.get_depth_frame(), aligned.get_color_frame()
        if not df or not cf:
            continue
        color = np.asanyarray(cf.get_data()); display = color.copy()
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        h, w, _ = color.shape

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(display, lm, mp_hands.HAND_CONNECTIONS)

            # # ---- 控制夹抓（原逻辑） ----
            # th = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
            # idx= lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # u_t, v_t = int(np.clip(int(th.x*w),0,w-1)), int(np.clip(int(th.y*h),0,h-1))
            # u_i, v_i = int(np.clip(int(idx.x*w),0,w-1)), int(np.clip(int(idx.y*h),0,h-1))
            # z_t = df.get_distance(u_t, v_t); z_i = df.get_distance(u_i, v_i)
            # if z_t and z_i and (not np.isnan(z_t)) and (not np.isnan(z_i)) and z_t>0.001 and z_i>0.001:
            #     x_t = (u_t - cx) * z_t / fx; y_t = (v_t - cy) * z_t / fy
            #     x_i = (u_i - cx) * z_i / fx; y_i = (v_i - cy) * z_i / fy
            #     d = np.linalg.norm([x_t-x_i, y_t-y_i, z_t-z_i])
            #     if d < 0.045:
            #         arm.rm_set_gripper_position(0, block=True, timeout=0)
            #     elif d > 0.055:
            #         arm.rm_set_gripper_position(1000, block=True, timeout=0)

            # ---- 用 MCP 集合计算当前位置（鲁棒） ----
            Pcam = get_stable_hand_pos_from_MCPs(lm, df, intr)
            if Pcam is None:
                # 如果连回退都没有有效深度，跳过这帧（保持旧命令）
                cv2.imshow("Hand→Arm Joint PBVS", display)
                if cv2.waitKey(1) == ord('q'): break
                continue

            # 位置增量（camera frame -> base frame）
            deltaPr = R_rc.dot(Pcam - Pc0) * SCALE
            target_pos = pose0 + deltaPr

            # 姿态（保持你原来的方法）
            Wc  = unproject_pixel(int(np.clip(int(lm.landmark[mp_hands.HandLandmark.WRIST].x*intr.width),0,intr.width-1)),
                                  int(np.clip(int(lm.landmark[mp_hands.HandLandmark.WRIST].y*intr.height),0,intr.height-1)),
                                  depth_median(df, int(np.clip(int(lm.landmark[mp_hands.HandLandmark.WRIST].x*intr.width),0,intr.width-1)),
                                                  int(np.clip(int(lm.landmark[mp_hands.HandLandmark.WRIST].y*intr.height),0,intr.height-1)), win=DEPTH_MEDIAN_WINDOW))
            Fc  = None
            Tc  = None
            try:
                Fc = unproject_pixel(int(lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x*intr.width),
                                     int(lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y*intr.height),
                                     depth_median(df,
                                                   int(lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x*intr.width),
                                                   int(lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y*intr.height),
                                                   win=DEPTH_MEDIAN_WINDOW))
                Tc = unproject_pixel(int(lm.landmark[mp_hands.HandLandmark.THUMB_MCP].x*intr.width),
                                     int(lm.landmark[mp_hands.HandLandmark.THUMB_MCP].y*intr.height),
                                     depth_median(df,
                                                   int(lm.landmark[mp_hands.HandLandmark.THUMB_MCP].x*intr.width),
                                                   int(lm.landmark[mp_hands.HandLandmark.THUMB_MCP].y*intr.height),
                                                   win=DEPTH_MEDIAN_WINDOW))
            except:
                pass

            if (Fc is None) or (Tc is None):
                # 若姿态点缺失，保留上次姿态（你也可以 skip 这帧）
                pass
            else:
                z_ax = normalize(Fc - Wc)
                tmp = Tc - Wc
                x_ax = normalize(tmp - np.dot(tmp, z_ax) * z_ax)
                y_ax = np.cross(z_ax, x_ax)
                R_hand = np.stack([x_ax, y_ax, z_ax], axis=1)
                R_base = R_rc.dot(R_hand)
                quat = rotation_matrix_to_quaternion(R_base)
                qv = np.array(quat, dtype=np.float64); qv /= np.linalg.norm(qv); quat = qv.tolist()
                eul = algo.rm_algo_quaternion2euler(quat)  # [rx,ry,rz]

            # ---- 位置平滑（EMA） & 跳变限幅 ----
            
            if pos_ema is None:
                pos_ema = target_pos.astype(np.float64)
            else:
                # EMA
                pos_ema = (1.0 - EMA_ALPHA_POS) * pos_ema + EMA_ALPHA_POS * target_pos
                # 限幅：如果跳变过大，限制最大帧步长
                # compute difference from last sent position: use last averaged pose in queue if exists
                last_sent = np.array(arm.rm_get_current_arm_state()[1]['pose'][:3], dtype=np.float64)
                step = pos_ema - last_sent
                step_norm = np.linalg.norm(step)    # 向量长度(距离)
                if step_norm > MAX_STEP_PER_FRAME:
                    step = step / step_norm * MAX_STEP_PER_FRAME
                    pos_ema = last_sent + step

            # 逆解并下发（欧拉作为姿态，保持你原来的接口）
            target_pose = [float(pos_ema[0]), float(pos_ema[1]), float(pos_ema[2]),
                           float(eul[0]), float(eul[1]), float(eul[2])]
            q_current = arm.rm_get_joint_degree()[1]
            params = rm_inverse_kinematics_params_t(q_current, target_pose, 1)
            ret_ik, joints = algo.rm_algo_inverse_kinematics(params)
            if ret_ik == 0:
                # 这里还用到了平均平滑
                joint_queue.append(joints)
                avg_joints = np.mean(joint_queue, axis=0).tolist()
                #print(avg_joints)
                arm.rm_movej_follow(avg_joints)
            else:
                if ret_ik == 1:
                    print("逆解失败")   # IK 失败则 skip（或你可以尝试不同 seed）
                else:
                    print("IK Code:", ret_ik)

        cv2.imshow("Hand→Arm Joint PBVS", display)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    hands.close()
    arm.rm_delete_robot_arm()
    cv2.destroyAllWindows()
