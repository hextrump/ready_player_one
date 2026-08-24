# lie_detector — MapleStory 测谎仪鼠标追踪模型

把 OpenCV 检测 + SAMURAI 跟踪打包成统一 facade (`LieDetectorModel`)，
方便 ready_player_one bot 和**其他机器整合代码**直接 import 即用。

## 包结构

```
src/perception/lie_detector/
├── __init__.py          # 公开 API: LieDetectorModel, LieDetectResult, LiePhase, LieBackend
├── model.py             # LieDetectorModel 统一 facade (去抖 + bbox 膨胀 + backend 切换 + 超时)
├── opencv_backend.py    # OpenCV 后端 (CPU, 默认; 复用以跑通的检测函数)
├── samurai_backend.py   # SAMURAI 后端 (GPU, 可选; 薄包装 samurai_track.py + sam2)
├── state.py             # LieDetectResult / LiePhase / LieBackend / 去抖 / bbox 膨胀
└── README.md            # 本文件
```

## 依赖（外部项目）

底层复用本地 `lie-detector/` 项目（独立项目，已跑通），**不复制代码**：

```
lie-detector/
├── scripts/
│   ├── auto_bbox.py             # detect_lie_detector_window + detect_white_target (多阈值)
│   ├── white_silhouette_detector.py  # 形状无关 mask + 模板匹配兜底
│   └── multiframe_bbox.py       # 1 秒 10 帧投票 (离线初始定位)
├── samurai_repo/
│   ├── sam2/                    # sam2.1 + checkpoints
│   └── scripts/                 # demo.py (SAMURAI 主入口)
└── samurai_track.py             # 一站式 (OpenCV init + SAMURAI propagate + bbox+center 输出)
```

bot 通过 `detector_repo_path` 指向该项目的绝对路径，运行时 `sys.path.insert` 注入。

## 最小用法

### 1. 实时集成（bot 视觉线程）

```python
from src.perception.lie_detector import LieDetectorModel, LieBackend
from src.utils.config import load_config

cfg = load_config()
ld_cfg = cfg.get("lie_detector", {})

model = LieDetectorModel(
    detector_repo_path=ld_cfg["detector_repo_path"],
    backend=ld_cfg.get("backend", "opencv"),
    config={
        "activate_after_frames": ld_cfg.get("activate_after_frames", 2),
        "deactivate_after_frames": ld_cfg.get("deactivate_after_frames", 6),
        "timeout_sec": ld_cfg.get("timeout_sec", 30.0),
    },
)

# 视觉线程每帧调用
while True:
    frame = capture.grab()           # BGR numpy, letterboxed
    result = model.update(frame)
    if result.active:
        # result.target_center = (cx, cy) in letterbox 帧坐标
        # result.confidence ∈ [0, 1] (多阈值命中数 / 总阈值数)
        # result.brightness ∈ [0, 255]
        # result.phase = COUNTDOWN | TRACKING
        mouse.move_to(result.target_center, result.confidence)
```

### 2. 离线分析（其他机器）

```python
import cv2
from src.perception.lie_detector import LieDetectorModel

model = LieDetectorModel(
    detector_repo_path="C:/path/to/lie-detector",
    backend="opencv",
)
cap = cv2.VideoCapture("clip.mp4")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    result = model.update(frame)
    if result.active:
        print(f"frame={int(cap.get(cv2.CAP_PROP_POS_FRAMES))} "
              f"center={result.target_center} conf={result.confidence:.2f}")
```

### 3. GPU 后端（SAMURAI）

需要：
- torch + CUDA（`torch.cuda.is_available() == True`）
- `lie-detector/samurai_repo/sam2/checkpoints/sam2.1_hiera_base_plus.pt`（或 large）

```python
model = LieDetectorModel(
    detector_repo_path="C:/path/to/lie-detector",
    backend="samurai",           # 不可用时自动降级 OpenCV
)
# 视频序列初始化: 把帧迭代器 + 首帧 bbox 喂给 samurai backend
# (具体流式接口见 samurai_backend.py 的 init_with_bbox + propagate)
```

## 坐标映射（视觉帧 → 屏幕）

视觉帧是 letterbox 后的尺寸（如 1366x768），鼠标需要屏幕坐标：

```python
scale, pad_left, pad_top = capture.last_letterbox        # WindowCapture 暴露
# 1. letterbox → client (原窗口客户区)
client_x = (target_center[0] - pad_left) / scale
client_y = (target_center[1] - pad_top) / scale
# 2. client → screen
import win32gui
client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))
screen_x = int(client_left + client_x)
screen_y = int(client_top + client_y)
# 3. 移动鼠标 (走 human_mouse 拟人化轨迹)
human_mouse.move_to(current_cursor, (screen_x, screen_y), duration_ms=120)
```

## 其他机器整合步骤

1. clone ready_player_one 仓库
2. clone（或拷贝）lie-detector 项目到本地
3. `pip install -r requirements.txt`（OpenCV 后端不需要 torch）
4. config.yaml 配 `lie_detector.detector_repo_path`
5. `from src.perception.lie_detector import LieDetectorModel` 即可

## 已知限制

- **沙色窗口检测**：实际游戏窗口是深灰/沙土浮雕，OpenCV 窗口检测常 FAILED → 自动回退全帧白块检测（实测工作）
- **SAMURAI 移动阶段跟丢**：初始 bbox 太紧 + 倒计时→移动窗口布局剧变会丢目标 → 用更大初始 bbox + OpenCV 重检出触发 re-init 缓解
- **目标渐隐后期精度**：OpenCV 后端在多阈值命中数降低时 bbox 漂移 → 配合 §9 自适应速度使用
- **CUDA 依赖**：SAMURAI 后端需 GPU；无 GPU 自动降级 OpenCV

详见 `design/测谎仪鼠标追踪.md` 设计文档。
