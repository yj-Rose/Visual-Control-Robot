#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# 相机 API
import pyrealsense2 as rs
# 机械臂 API
from Robotic_Arm.rm_robot_interface import *

# =========================================================
# 0. 配置
# =========================================================
# - eye_in_hand : 相机在末端/法兰上，标定板固定在外部世界（桌面/工装）
#   输出: T_cam2gripper (= T_cam2flange)
#
# - eye_to_hand : 相机固定在外部，标定板固定在末端/法兰上
#   输出: T_cam2base 以及 T_base2cam
SETUP_MODE = "eye_to_hand"   # "eye_in_hand 眼在手上" or "eye_to_hand 眼在手外"

# 采样质量控制
MIN_CHARUCO_CORNERS = 6    # 最小角点
MIN_SAMPLES = 10           # 最少样本
MAX_REPROJ_ERR_PX = 2.5    # 重投影误差

# =========================================================
# 1. RealMan 机械臂初始化      ！！！ 需要换成你们的机械臂配置通讯
# =========================================================
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)

# =========================================================
# 2. RealSense 初始化         ！！！ 需要换成你们的奥比中光相机
# =========================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# =========================================================
# 3. 相机内参（直接来自 RealSense）  ！！！realsense相机出厂标定好内参了可以直接读到，你们可以查一下奥比中光的内参能不能直接读到，如果不能自己标定内参手动填入K就行
# =========================================================
profile = pipeline.get_active_profile()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()

K = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)

D = np.array(intr.coeffs, dtype=np.float64)

# =========================================================
# 4. ChArUco 板定义（必须与你打印的一致）
# =========================================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

squares_x = 5
squares_y = 7
square_length = 0.0363   # 米
marker_length = 0.027   # 米

board = aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    aruco_dict
)

detector = aruco.CharucoDetector(board)

# =========================================================
# 5. 数据容器
# =========================================================
R_gripper2base = []   # ^bT_g : gripper(frame g) -> base(frame b) 夹爪 → 基座 读机械臂末端位姿
t_gripper2base = []

R_target2cam = []     # P -> C 标定板 → 相机（PnP）
t_target2cam = []

print("\n==============================")
print(f"手眼标定模式: {SETUP_MODE}")
print("按 c 采集一组数据")
print("按 q 结束并计算")
print("==============================\n")

def rm_matrix_to_numpy(mat: rm_matrix_t) -> np.ndarray:
    """rm_matrix_t(4x4) -> numpy(4,4), 假设数据为行优先展开。"""
    data = np.array(list(mat.data), dtype=np.float64).reshape(4, 4)
    return data

def invert_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    tm = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3:4] = -Rm.T @ tm
    return Ti

def reprojection_error_px(obj_pts: np.ndarray, img_pts: np.ndarray,
                          rvec: np.ndarray, tvec: np.ndarray,
                          K: np.ndarray, D: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    img = img_pts.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img, axis=1)))

# =========================================================
# 6. 主循环：采集数据
# =========================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        ok = False
        rvec = None
        tvec = None
        cur_reproj = None

        if charuco_ids is not None and len(charuco_ids) >= MIN_CHARUCO_CORNERS:
            # -------- 正确构造 PnP 对应 --------
            obj_pts = []
            img_pts = []

            chessboard_corners = board.getChessboardCorners()

            for i, cid in enumerate(charuco_ids.flatten()):
                obj_pts.append(chessboard_corners[cid])
                img_pts.append(charuco_corners[i][0])

            obj_pts = np.asarray(obj_pts, dtype=np.float64)
            img_pts = np.asarray(img_pts, dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                K,
                D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if ok:
                cur_reproj = reprojection_error_px(obj_pts, img_pts, rvec, tvec, K, D)
                cv2.drawFrameAxes(img, K, D, rvec, tvec, 0.05)
                cv2.putText(
                    img,
                    f"reproj={cur_reproj:.2f}px  corners={len(charuco_ids)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if (cur_reproj is not None and cur_reproj <= MAX_REPROJ_ERR_PX) else (0, 0, 255),
                    2,
                )

        cv2.imshow("hand_eye_calibration", img)
        key = cv2.waitKey(1) & 0xFF

        # -----------------------------------------------------
        # 采集一组
        # -----------------------------------------------------
        if key == ord('c') and (charuco_ids is not None) and ok:
            if cur_reproj is None or cur_reproj > MAX_REPROJ_ERR_PX:
                print(f"✘ 拒绝采样：PnP 重投影误差过大 ({cur_reproj:.2f}px > {MAX_REPROJ_ERR_PX}px)")
                continue

            # ---- target(P) -> camera(C)
            R_tc, _ = cv2.Rodrigues(rvec)
            t_tc = tvec.reshape(3, 1)

            R_target2cam.append(R_tc)
            t_target2cam.append(t_tc)

            #  gripper(g) -> base(b): 从机器人库直接拿 4x4，避免欧拉顺序/表达误用
            # ========================================================
            state = arm.rm_get_current_arm_state()[1]    # 这是realman读末端姿态的API 换成你们的机械臂.
            pose = state['pose']  # [x,y,z,rx,ry,rz] 位置:m  欧拉角:rad
            T_bg = rm_matrix_to_numpy(arm.rm_algo_pos2matrix(pose))  # ^bT_g  realman的API 把位姿转换成矩阵形式
            # ==============================================================

            R_gripper2base.append(T_bg[:3, :3].copy())
            t_gripper2base.append(T_bg[:3, 3:4].copy())

            print(f"✔ 已采集 {len(R_gripper2base)} 组 (reproj={cur_reproj:.2f}px)")

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    arm.rm_delete_robot_arm()

# =========================================================
# 7. Hand-Eye 标定
# =========================================================
if len(R_gripper2base) < MIN_SAMPLES:
    raise RuntimeError(f"采样不足：当前 {len(R_gripper2base)} 组，建议至少 {MIN_SAMPLES} 组（且姿态变化要足够大）")

np.set_printoptions(precision=6, suppress=True)

if SETUP_MODE == "eye_in_hand":
    # OpenCV 定义：输入 ^bT_g (gripper->base) 与 ^cT_t (target->cam)，输出 ^gT_c (cam->gripper)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2gripper = np.eye(4, dtype=np.float64)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

    rot = R.from_matrix(R_cam2gripper)
    euler_deg = rot.as_euler('zyx', degrees=True)
    quat_xyzw = rot.as_quat()

    print("\n================= 标定结果 =================")
    print("【T_cam2gripper / T_cam2flange】 (^gT_c)")
    print(T_cam2gripper)
    print("\n平移(米):", t_cam2gripper.flatten())
    print("欧拉角 ZYX(deg):", euler_deg)
    print("四元数 xyzw:", quat_xyzw)
    print("===========================================\n")

elif SETUP_MODE == "eye_to_hand":
    # OpenCV 文档给出的 eye-to-hand 形式需要用 ^gT_b（base->gripper 的“逆”），解出 ^bT_c（cam->base）
    R_gTb = []
    t_gTb = []
    for R_bTg, t_bTg in zip(R_gripper2base, t_gripper2base):
        # bTg -> gTb
        R_ = R_bTg.T            # .T 是转置的意思
        t_ = -R_bTg.T @ t_bTg
        R_gTb.append(R_)
        t_gTb.append(t_)

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gTb,
        t_gTb,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2base = np.eye(4, dtype=np.float64)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()
    # T_base2cam = invert_T(T_cam2base)

    rot = R.from_matrix(R_cam2base)
    euler_deg = rot.as_euler('zyx', degrees=True)
    quat_xyzw = rot.as_quat()

    print("\n================= 标定结果 =================")
    print("【T_cam2base】 (^bT_c)")
    print(T_cam2base)
    # print("\n【T_base2cam】 (^cT_b)")
    # print(T_base2cam)
    print("\n平移(米):", t_cam2base.flatten())
    print("欧拉角 ZYX(deg):", euler_deg)
    print("四元数 xyzw:", quat_xyzw)
    print("===========================================\n")
else:
    raise ValueError(f"未知 SETUP_MODE={SETUP_MODE}, 只能是 'eye_in_hand' 或 'eye_to_hand'")
