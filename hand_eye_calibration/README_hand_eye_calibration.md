# 手眼标定脚本适配指南（机械臂 + 奥比中光/其他相机）

该脚本当前已适配 **RealMan 机械臂** + **Intel RealSense 彩色相机**，并使用 **ChArUco** 标定板进行手眼标定。  
如果“使用其他机械臂 + 奥比中光（Orbbec）相机”需要改动指定代码段。

---

### 1. 这个脚本在做什么

- **输入数据（每采样一次）**：
  - 机械臂末端相对基座的位姿：\({}^bT_g\)（gripper/法兰 -> base）
  - 标定板相对相机的位姿：\({}^cT_t\)（target/board -> camera），由图像 + 相机内参通过 PnP 求得
- **输出**：
  - `SETUP_MODE="eye_in_hand"`（眼在手上）：输出 \({}^gT_c\)（camera -> gripper/法兰），脚本里打印为 `T_cam2gripper`
  - `SETUP_MODE="eye_to_hand"`（眼在手外）：输出 \({}^bT_c\)（camera -> base），脚本里打印为 `T_cam2base`

脚本的核心 OpenCV 调用是 `cv2.calibrateHandEye(...)`，其前提是：你提供给它的**相机位姿**与**机械臂位姿**必须在同一套物理场景下严格同步采集，且单位/坐标系定义一致。

---

### 2. 两种安装模式与坐标系约定（非常重要）

脚本通过 `SETUP_MODE` 选择模式：

# - eye_in_hand : 相机在末端/法兰上，标定板固定在外部世界（桌面/工装）
#   输出: T_cam2gripper (= T_cam2flange)
#
# - eye_to_hand : 相机固定在外部，标定板固定在末端/法兰上
#   输出: T_cam2base 以及 T_base2cam
SETUP_MODE = "eye_to_hand"   # "eye_in_hand 眼在手上" or "eye_to_hand 眼在手外"
```

- **base（b）**：机械臂基座坐标系（机器人本体定义）
- **gripper（g）**：末端执行器/法兰坐标系（机器人本体定义）
- **camera（c）**：相机坐标系（OpenCV 约定：Z 向前、X 向右、Y 向下）
- **target/board（t）**：标定板坐标系（ChArUco 板在 OpenCV 中的模型坐标）

你需要保证：
- 机械臂部分提供的是 **\({}^bT_g\)**（gripper->base）的 4x4 齐次矩阵（或等价的 R/t）
- 相机部分通过 PnP 求得的是 **\({}^cT_t\)**（target->camera）的 R/t

---

### 3. 你需要改哪些地方（总结）

必须改的通常只有三块：

- **(A) 机械臂 API**：连接方式、获取末端位姿、把位姿转换为 4x4 齐次矩阵  
  对应脚本第 1 段初始化 + 采样时读取位姿（见第 5 节）
- **(B) 相机 API（Orbbec）**：取彩色图、相机内参 \(K\) 和畸变 \(D\) 的来源  
  对应脚本第 2、3 段（见第 4 节）
- **(C) ChArUco 板参数**：棋盘格尺寸、marker 尺寸、字典类型必须与你打印的板一致  
  对应脚本第 4 段（见第 6 节）

其余部分（PnP、重投影误差筛选、calibrateHandEye 调用）通常不需要改。

---

### 4. 适配相机（把 RealSense 换成奥比中光/其他相机）

当前脚本使用 RealSense SDK：

# 相机 API
import pyrealsense2 as rs
...
# 2. RealSense 初始化         ！！！ 需要换成你们的奥比中光相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
...
# 3. 相机内参（直接来自 RealSense）  ！！！realsense相机出厂标定好内参了可以直接读到...
profile = pipeline.get_active_profile()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()

K = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)

D = np.array(intr.coeffs, dtype=np.float64)
```

你要做的是：**让脚本最终拿到两样东西**：
- **`img`**：一张 `numpy.ndarray` 的 BGR 图（形状 HxWx3，dtype=uint8）
- **`K, D`**：相机内参矩阵与畸变系数（OpenCV 格式）

#### 4.1 替换相机取流（产出 `img`）

脚本在主循环中使用：

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
...
img = np.asanyarray(color_frame.get_data())
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

适配 Orbbec 的目标是把以上逻辑替换成“你们相机 SDK/驱动”的读取方式，并保证 `img` 是 BGR。

建议改法：封装一个 `get_color_image_bgr()`，主循环里只调用它：

```python
def get_color_image_bgr() -> np.ndarray:
    """
    返回一帧 BGR 图：HxWx3, uint8
    - 如果你的 SDK 输出是 RGB，需要 cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    - 如果输出是 YUYV/NV12 等，需要先转换到 BGR
    """
    raise NotImplementedError
```

然后把主循环中 `img = ...` 替换为 `img = get_color_image_bgr()`。

#### 4.2 获取/配置相机内参（产出 `K` 和 `D`）

PnP 会强依赖内参：如果 \(K/D\) 错，手眼结果一定漂。

你有三种常见来源（任选其一）：

- **(1) 直接从 Orbbec SDK 读取**：很多型号可读到 fx/fy/cx/cy 及畸变参数  
- **(2) 通过 ROS/驱动发布的 CameraInfo**：拿到 `K` 和 `D` 
- **(3) 自己做一次相机标定**：用棋盘格/ChArUco 标定得到 `K`、`D`

无论来源是什么，你最终需要在脚本里得到 OpenCV 兼容格式：

```python
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# 常见畸变：k1,k2,p1,p2,k3（也可能更多，如 k4,k5,k6）
D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
```

**注意事项**：
- **分辨率必须一致**：你读图像的分辨率（如 1280x720）必须对应同一套内参；如果改了分辨率，需要用对应分辨率的内参（或按相机模型正确缩放）
- **单位无关**：`K` 是像素单位；`D` 是无量纲
- **畸变为空也能跑**：若你没有畸变参数，可先用 `D=np.zeros((5,), dtype=np.float64)` 跑通流程，但结果可能不如真实畸变好

---

### 5. 适配机械臂（把 RealMan 换成你们的机械臂）

当前脚本使用 RealMan API 初始化并读取位姿：

```
# 1. RealMan 机械臂初始化      ！！！ 需要换成你们的机械臂配置通讯
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
```

采样时读取末端位姿并转成矩阵：

```
#  gripper(g) -> base(b): 从机器人库直接拿 4x4，避免欧拉顺序/表达误用
state = arm.rm_get_current_arm_state()[1]    # 这是realman读末端姿态的API 换成你们的机械臂.
pose = state['pose']  # [x,y,z,rx,ry,rz] 位置:m  欧拉角:rad
T_bg = rm_matrix_to_numpy(arm.rm_algo_pos2matrix(pose))  # ^bT_g  realman的API 把位姿转换成矩阵形式

R_gripper2base.append(T_bg[:3, :3].copy())
t_gripper2base.append(T_bg[:3, 3:4].copy())
```

你要做的是：**在每次按下 `c` 采样时，得到一帧对应的 \({}^bT_g\)**，并填入：
- `R_gripper2base.append(T_bg[:3, :3])`
- `t_gripper2base.append(T_bg[:3, 3:4])`

#### 5.1 你们的机械臂需要提供什么数据

至少需要下面任意一种：

- **(推荐) 直接给 4x4 齐次矩阵 \({}^bT_g\)**（单位米）
- 或给 **位置 + 姿态**，你自己在代码里转成矩阵
  - 姿态可以是：旋转矩阵 / 四元数 / 轴角 / 欧拉角（但欧拉角最容易踩坑）

#### 5.2 欧拉角/旋转约定的坑（请务必核对）

如果你们的机械臂只提供欧拉角（rx,ry,rz），你必须确认：
- 欧拉顺序（XYZ / ZYX / …）
- 是**内旋**还是**外旋**
- 角度单位（deg / rad）
- 坐标系方向（右手/左手）

只要其中任何一项弄错，手眼结果会明显不稳定或完全错误。  
因此建议：**尽量让机器人 SDK 直接输出 4x4**（或直接输出旋转矩阵）。

#### 5.3 建议的“机械臂适配层”写法

把所有机械臂相关代码集中到两个函数，其他逻辑不动：

```python
def robot_init():
    """连接你们的机械臂，返回一个句柄/对象"""
    raise NotImplementedError

def robot_get_T_bg(robot) -> np.ndarray:
    """
    返回 ^bT_g (4x4)：
    - 平移单位：米
    - 旋转：3x3 正交矩阵
    """
    raise NotImplementedError
```

然后在采样处替换为：

```python
T_bg = robot_get_T_bg(robot)
R_gripper2base.append(T_bg[:3, :3].copy())
t_gripper2base.append(T_bg[:3, 3:4].copy())
```

---

### 6. 适配 ChArUco 标定板（必须与打印一致）

脚本当前的 ChArUco 板定义如下：

```58:74:realman/hand_eye_calibration.py
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

squares_x = 5
squares_y = 7
square_length = 0.04   # 米
marker_length = 0.03   # 米

board = aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    aruco_dict
)

detector = aruco.CharucoDetector(board)
```

你必须让以下参数与实际打印板一致：
- **`aruco_dict`**：字典类型（例如 `DICT_6X6_250`）
- **`squares_x/squares_y`**：棋盘格数量
- **`square_length`**：棋盘格边长（单位：米）
- **`marker_length`**：Aruco marker 边长（单位：米）

---

### 7. 采样与质量控制（不改）

脚本用以下阈值做数据质量筛选：

```23:27:realman/hand_eye_calibration.py
MIN_CHARUCO_CORNERS = 6    # 最小角点
MIN_SAMPLES = 10           # 最少样本
MAX_REPROJ_ERR_PX = 2.5    # 重投影误差
```

建议：
- **样本数**：至少 10 组，实际建议 20~40 组
- **姿态变化**：采样时要让末端产生“明显的旋转变化”（不是只平移），否则可观测性差
- **重投影误差**：一般 1~3px 合理；如果你们相机噪声大/分辨率低，可适当放宽，但不要太大

---

### 8. 运行方式（给对方的操作步骤）

1. 安装依赖（至少需要）：
   - Python3
   - OpenCV（带 aruco 模块，通常是 `opencv-contrib-python`）
   - numpy
   - scipy
   - 你们相机 SDK 的 Python 绑定（Orbbec SDK/OpenNI/ROS 等）
   - 你们机械臂 SDK 的 Python 包
2. 根据第 4/5/6 节完成相机、机械臂、标定板参数适配
3. 运行脚本后：
   - 相机画面窗口出现
   - 按 `c` 采集一组数据（屏幕会显示 reproj=xx px）
   - 按 `q` 结束并输出结果矩阵

---

### 9. 验证

建议按以下顺序排查：

- **(1) PnP 是否稳定**：窗口里 `reproj=...px` 是否稳定在较小范围（例如 0.5~2.5px），轴是否贴在板上且方向合理
- **(2) 机械臂位姿单位是否正确**：如果你们的位姿是 mm，需要除以 1000 转成 m（否则平移会大 1000 倍）
- **(3) 结果是否符合直觉**：
  - eye-in-hand：`T_cam2gripper` 的平移大致等于相机到法兰的实际安装偏置
  - eye-to-hand：`T_cam2base` 的平移大致等于相机在工作空间中的安装位置（相对 base）
- **(4) 重复性测试**：用两次独立采样计算的结果是否接近（差异不应很夸张）

---

### 10. 常见错误

- **相机内参与分辨率不匹配**：改了取流分辨率却没换内参
- **把 RGB 当 BGR**：颜色不影响 PnP，但若你中间做了阈值/预处理会出问题；建议统一 BGR
- **机械臂位姿不是 \({}^bT_g\)**：有些 SDK 给的是 \({}^gT_b\) 或 TCP->base 的逆，需要确认
- **欧拉角顺序错**：尤其是 (rx,ry,rz) 的定义不清
- **平移单位错误**：mm 当 m，或反过来
- **采样姿态变化太小**：几乎同姿态只平移，容易导致解不稳定

---

### 11. 输出结果怎么用（简单说明）

- 若 `SETUP_MODE="eye_in_hand"`，你得到的是 \({}^gT_c\)：把相机坐标下的点/位姿转换到 gripper/法兰坐标可用
- 若 `SETUP_MODE="eye_to_hand"`，你得到的是 \({}^bT_c\)：把相机坐标下的点/位姿转换到机器人 base 坐标可用

如果你还需要 `T_base2cam`（\({}^cT_b\)），脚本里已经有 `invert_T()`，可以直接把注释打开：

```96:102:realman/hand_eye_calibration.py
def invert_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    tm = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3:4] = -Rm.T @ tm
    return Ti
```


