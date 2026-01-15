"""
OpenCV webcam -> MMPose hand 2D keypoints (21) demo.

目标：像 `opencv_mediapipe.py` 一样，实时检测手部关键点并在 OpenCV 窗口里画点/连线。

运行示例（推荐在你配置好的 openmmlab conda 环境里）：
  python /home/linkeros/Rose/realman/VisualControl/test/opencv_mmpose_hand.py --cam-id 4

如果你的环境没有安装 MMDetection（mmdet），top-down 模型会无法自动初始化检测器；
此时可以用 whole-image 模式先跑通（适合手部占画面较大/单手）：
  python .../opencv_mmpose_hand.py --det-model whole_image
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _try_add_local_mmpose_to_syspath() -> None:
    """允许直接用本仓库的 `realman/mmpose`（即使没 pip install）。"""
    this_file = os.path.abspath(__file__)
    # /home/linkeros/Rose/realman/VisualControl/test/opencv_mmpose_hand.py
    # -> /home/linkeros/Rose/realman/mmpose
    repo_root = os.path.abspath(os.path.join(os.path.dirname(this_file), "..", "..", "mmpose"))
    if os.path.isdir(repo_root) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_try_add_local_mmpose_to_syspath()

# noqa: E402
from mmpose.apis import MMPoseInferencer  # type: ignore
from mmpose.utils import register_all_modules  # type: ignore


# 21 点手骨架连接（与常见的 MediaPipe hand 21 点拓扑一致）
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(
    img_bgr: np.ndarray,
    kpts_xy: Sequence[Sequence[float]],
    kpt_scores: Sequence[float] | None = None,
    kpt_thr: float = 0.3,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 4,
    thickness: int = 2,
) -> None:
    """在 BGR 图上画关键点与骨架。"""
    if kpt_scores is None:
        kpt_scores = [1.0] * len(kpts_xy)

    # 画连线
    for a, b in HAND_CONNECTIONS:
        if a >= len(kpts_xy) or b >= len(kpts_xy):
            continue
        if kpt_scores[a] < kpt_thr or kpt_scores[b] < kpt_thr:
            continue
        xa, ya = int(kpts_xy[a][0]), int(kpts_xy[a][1])
        xb, yb = int(kpts_xy[b][0]), int(kpts_xy[b][1])
        cv2.line(img_bgr, (xa, ya), (xb, yb), color, thickness, lineType=cv2.LINE_AA)

    # 画点
    for i, (x, y) in enumerate(kpts_xy):
        if i < len(kpt_scores) and kpt_scores[i] < kpt_thr:
            continue
        cv2.circle(img_bgr, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    default_cfg = "/home/linkeros/Rose/realman/mmpose/configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py"
    default_ckpt = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth"
    )

    p = argparse.ArgumentParser()
    p.add_argument("--cam-id", type=int, default=4, help="摄像头设备 ID（与你现在 mediapipe 脚本一致默认 4）")
    p.add_argument("--flip", action="store_true", help="是否水平镜像（更符合自拍交互）")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="关键点可视化阈值（0~1）")
    p.add_argument("--num-instances", type=int, default=1, help="最多显示几只手（按置信度排序）")

    p.add_argument("--pose2d", type=str, default=default_cfg, help="手部 2D 姿态模型 config 路径/别名/配置名")
    p.add_argument("--pose2d-weights", type=str, default=default_ckpt, help="手部 2D 姿态模型权重（路径或 URL）")
    p.add_argument("--device", type=str, default="cuda:0", help="推理设备，如 cuda:0 / cpu；默认让 MMPose 自动选择")

    # det_model:
    # - None: 让 MMPose 自动按数据集类型选默认 detector（需要 mmdet）
    # - whole_image: 不用 detector，整张图当一个 bbox（手要够大才稳）
    p.add_argument("--det-model", type=str, default=None, help="检测器：None(自动) / whole_image / mmdet config/别名")
    p.add_argument("--det-weights", type=str, default=None, help="检测器权重（路径或 URL）")
    p.add_argument("--det-cat-ids", type=int, nargs="*", default=None, help="检测类别 id（通常不需要手动填）")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    register_all_modules(init_default_scope=True)

    # 初始化 inferencer
    # - 如果 det_model=None 且你装了 mmdet，会自动初始化 hand detector
    # - 如果没装 mmdet，会抛 RuntimeError；我们退化到 whole_image 先跑通
    try:
        inferencer = MMPoseInferencer(
            pose2d=args.pose2d,
            pose2d_weights=args.pose2d_weights,
            device=args.device,
            det_model=args.det_model,
            det_weights=args.det_weights,
            det_cat_ids=args.det_cat_ids,
        )
    except RuntimeError as e:
        msg = str(e)
        if "MMDetection" in msg or "mmdet" in msg:
            print("⚠️ 检测到当前环境可能未安装 MMDetection（mmdet）。")
            print("   将自动切换到 --det-model whole_image（整图 bbox）模式先跑通。")
            inferencer = MMPoseInferencer(
                pose2d=args.pose2d,
                pose2d_weights=args.pose2d_weights,
                device=args.device,
                det_model="whole_image",
            )
        else:
            raise

    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头：cam_id={args.cam_id}")

    win = "MMPose Hand Tracking"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_t = time.time()
    fps_ema = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # MMPoseInferencer 支持 numpy array 输入
            result = next(inferencer(frame))
            preds = result.get("predictions", [[]])
            instances = preds[0] if len(preds) > 0 else []

            # 画关键点（最多 num_instances）
            shown = 0
            for inst in instances:
                if shown >= args.num_instances:
                    break

                kpts = inst.get("keypoints", None)
                scores = inst.get("keypoint_scores", None)
                if kpts is None:
                    continue
                if len(kpts) != 21:
                    # 你换了别的数据集/模型时可能不是 21 点，这里先跳过，避免画错拓扑
                    continue

                draw_hand(
                    frame,
                    kpts,
                    scores,
                    kpt_thr=args.kpt_thr,
                    color=(0, 255, 0),
                )
                shown += 1

            # FPS 显示（EMA）
            now = time.time()
            inst_fps = 1.0 / max(1e-6, (now - last_t))
            last_t = now
            fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)

            cv2.putText(
                frame,
                f"hands={min(len(instances), args.num_instances)}  fps={fps_ema:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


