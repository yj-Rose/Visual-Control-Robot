# VisualControl：基于视觉的人手遥操作机械臂（PBVS / 增量映射）

本仓库展示一个**通过视觉识别手部姿态来遥操作机械臂**的原型项目：使用深度相机获取手部关键点的 3D 位置，在手上建立局部坐标系并映射到机械臂末端控制姿态；位置采用 **delta translation（增量位移）** 方式控制，核心依赖 **手眼标定**得到相机坐标系与机器人坐标系之间的旋转矩阵（以及可选的平移）。

> 适用场景：交互式示教、远程抓取、直觉式末端对齐等。  
> 注意：这是实验性质代码，真实机器人请务必加速度/空间限位/急停等安全约束。

---

## 项目核心思想

在相机坐标系下从手部关键点构建“手的坐标系”，并通过手眼标定得到的外参把它映射到机器人基座坐标系，从而将人的手势/位移转化为机械臂末端的**姿态 + 增量位置**控制命令。

---

## 坐标系约定与整体链路

常用坐标系：

- **\(\{C\}\)**：相机坐标系（OpenCV/RealSense 常见约定：Z 向前）
- **\(\{B\}\)**：机器人基座坐标系
- **\(\{E\}\)**：机器人末端（法兰/TCP）坐标系
- **\(\{H\}\)**：手部局部坐标系（由手腕/指根等关键点构建）

整体链路：

1. **手部2D关键点**：用 MediaPipe Hands（或 MMPose）检测 21 个关键点。
2. **深度反投影得到 3D 点**：利用深度图 + 相机内参 \(K\) 把像素点反投影为 \(\{C\}\) 下的 3D 坐标。
3. **手部坐标系 \(\{H\}\) 构建**：用 3 个（或更多）关键点构建正交基得到 \(R_{CH}\)。
4. **手眼标定得到 \(\{C\}\to\{B\}\) 的旋转**：得到 \(R_{BC}\)（本文中代码里常写成 `R_rc`）。
5. **映射到机器人**：
   - 位置：增量位移 \( \Delta p_B = R_{BC}\,\Delta p_C \)
   - 姿态：\( R_{BH} = R_{BC}\,R_{CH} \)

---

## 关键算法细节

### 1) 深度反投影（像素 \(\to\) 相机系 3D）

对某个关键点像素 \((u,v)\)，深度为 \(D\)（单位 m），相机内参为 \(f_x,f_y,c_x,c_y\)，则相机系 3D：

\[
X=\frac{(u-c_x)D}{f_x},\quad
Y=\frac{(v-c_y)D}{f_y},\quad
Z=D
\]

仓库实现示例见：`arm_control/test1.py` 中 `unproject()`。

### 2) 在手上建立坐标系（构建姿态）

一个常用构造（你在代码里已经用到了）：

- 取 **手腕** \(W\)、**中指根** \(F\)、**拇指根** \(T\) 的 3D 点（都在 \(\{C\}\)）
- 定义：
  - \( \hat{z} = \mathrm{normalize}(F-W) \)（腕 \(\to\) 中指根）
  - \( \hat{x} = \mathrm{normalize}\big((T-W) - ((T-W)\cdot \hat{z})\hat{z}\big) \)（腕 \(\to\) 拇指根，在 \(\hat{z}\) 上正交投影后归一化）
  - \( \hat{y} = \hat{z} \times \hat{x} \)

组成旋转矩阵（列向量为各轴）：

\[
R_{CH} = [\hat{x}\ \hat{y}\ \hat{z}]
\]

然后通过手眼旋转把手姿态映射到机器人基座：

\[
R_{BH}=R_{BC}R_{CH}
\]

最终可转换为四元数/欧拉角发送给机械臂逆解或跟随接口。

### 3) 位置控制：delta translation（增量位移映射）

相比“绝对位置映射”，本项目采用**增量映射**更稳定、也更符合遥操作习惯：

- 在启动阶段做“零位标定”，记录手腕（或 MCP 点集合）的相机系 3D 位置 \(p_{C0}\)
- 每帧计算当前手位置 \(p_C\)
- 相机系增量：

\[
\Delta p_C = p_C - p_{C0}
\]

- 映射到机器人基座系：

\[
\Delta p_B = s \cdot R_{BC}\,\Delta p_C
\]

其中 \(s\) 是映射灵敏度（代码里 `scale` / `SCALE`）。

最终目标末端位置：

\[
p_{B,\text{target}} = p_{B0} + \Delta p_B
\]

`arm_control/test1.py`/`test3.py` 中即为 `target_pos = pose0 + deltaPr`。

---

## 手眼标定：为什么需要、输出是什么、如何用

**为什么需要手眼标定？**  
因为手的 3D 位移/姿态最初是在相机坐标系 \(\{C\}\) 下得到的，而机器人控制通常在 \(\{B\}\)（基座）或 \(\{E\}\)（末端）下进行。要把“相机里看到的运动”变成“机器人坐标系里的运动”，必须知道 \(\{C\}\) 与 \(\{B\}\)/\(\{E\}\) 的相对位姿，尤其是**旋转矩阵**。

本仓库的手眼标定脚本：`hand_eye_calibration/hand_eye_calibration.py`  
说明文档：`hand_eye_calibration/README_hand_eye_calibration.md`

### 标定模式

- **eye-to-hand（眼在手外）**：相机固定在外部，标定板安装在末端。输出常用为 \({}^bT_c\)（camera \(\to\) base）。
- **eye-in-hand（眼在手上）**：相机安装在末端，标定板固定在外部。输出常用为 \({}^gT_c\)（camera \(\to\) gripper/flange）。

脚本通过 `SETUP_MODE` 选择，并基于 `cv2.calibrateHandEye(...)` 计算外参。

### 在遥操作里怎么用

你当前的遥操作代码中使用了一个旋转矩阵 `R_rc`（名字含义可以理解为 \(R_{BC}\) 或 \(R_{RB}\) 这类“把相机系向量旋到机器人基座系”的旋转）。实践中建议：

- **从手眼标定结果的 4x4 矩阵里取旋转块**作为 \(R_{BC}\)
- 如果你只需要“方向与位移映射”，只用旋转也能跑通；但要做更精确的“绝对位姿映射/抓取定位”，建议同时使用平移。

一个最直接的替换方式（以 eye-to-hand 输出的 `T_cam2base` 为例）：

```python
# T_cam2base 是 4x4：把 {C} 下的点/向量变换到 {B}
R_rc = T_cam2base[:3, :3].astype(np.float32)  # 作为 R_BC 使用

# 然后你现有的映射保持不变：
# deltaPr = R_rc.dot(deltaPc) * scale
# R_base  = R_rc.dot(R_hand)
```

> 注意：如果你拿到的是 `T_base2cam`（方向相反），那就需要取逆（旋转块用转置即可）：`R_rc = R_cb.T`。

---

## 快速开始

### 1) 硬件

- 一台深度相机（仓库当前示例为 Intel RealSense）
- 机械臂（仓库当前示例为 RealMan SDK；其他机械臂需适配接口）
- 一块 ChArUco 标定板（见 `hand_eye_calibration/generate_calibration_board/`）

### 2) 环境依赖（示例）

本项目未做统一的 `requirements.txt`，你可以按使用的脚本安装对应依赖（常见组合）：

- `numpy`
- `opencv-contrib-python`（需要 aruco 模块用于 ChArUco）
- `mediapipe`
- `scipy`
- `pyrealsense2`（如果使用 RealSense）
- 你的机械臂 SDK Python 包（如 RealMan 的 `Robotic_Arm`）

### 3) 手眼标定

运行：

- `python hand_eye_calibration/hand_eye_calibration.py`

按窗口提示采集数据并得到 `T_cam2base` 或 `T_cam2gripper`，把其中的旋转矩阵（或换算后的四元数）填入遥操作脚本里（目前 `arm_control/test1.py` 等脚本里是“手动硬编码四元数”）。

### 4) 运行手部遥操作（示例脚本）

- **主版本（位置增量 + 姿态 + 逆解 + 关节跟随）**：`arm_control/test1.py`
- **笛卡尔 PBVS（仅位置增量/姿态可选）**：`arm_control/test3.py`
- **抗抖/限幅等实验版本**：`arm_control/test2.py`

---

## 目录结构

- `arm_control/`
  - `test1.py`：MediaPipe + RealSense 深度反投影 + 手坐标系姿态 + delta trans + IK + `rm_movej_follow`
  - `test2.py`：在 `test1` 基础上增加更强的稳健性（多点深度、中值/EMA、关节限幅等）
  - `test3.py`：PBVS 增量映射的修改测试/实验版本
  - `mediapipe_robot.ipynb`：若干实验记录与可视化
- `hand_eye_calibration/`
  - `hand_eye_calibration.py`：ChArUco + PnP + OpenCV 手眼标定（支持 eye-to-hand / eye-in-hand）
  - `README_hand_eye_calibration.md`：适配不同机械臂/相机的详细说明
  - `generate_calibration_board/`：生成 ChArUco 标定板
- `hand_recognition_test/`
  - `opencv_mediapipe.py`：MediaPipe 手部检测可视化
  - `opencv_mmpose_hand.py`：MMPose 手部关键点可视化（可替换 MediaPipe）

---

## 一些经验与“坑”

- **必须确认坐标系方向与旋转乘法顺序**：你现在用的是 `R_base = R_rc.dot(R_hand)`，意味着 `R_rc` 把 \(\{C\}\) 下向量旋到 \(\{B\}\)。如果手眼标定输出矩阵方向不同，需要取逆或转置。
- **建议用增量映射而非绝对映射**：增量天然抗漂移、也更不依赖精确平移外参。
- **深度噪声会直接导致抖动**：可用邻域中值、EMA、卡尔曼、关节空间滑动平均、每帧限幅等（`arm_control/test2.py` 有示例）。
- **逆解跳解/奇异点**：建议用“上一帧下发的关节角”作为 IK seed，并对关节步长限幅；必要时做奇异性分析并跳过危险解（`mediapipe_robot.ipynb` 有相关实验）。
- **安全**：务必加工作空间约束、速度/加速度限制、碰撞/力控保护、急停按钮；

---



