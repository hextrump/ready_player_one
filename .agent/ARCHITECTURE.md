# 🌌 Project Architecture: Ready Player One (v3.1)

> 游戏自动化 bot. 核心是 **Super Brain** 单一 YOLO 模型, 一致输出 V13 4 类对象.

## 🎯 核心原则: Single Model Principle

一个 YOLO 模型检测游戏中所有关键对象, 输出 V13 四类 (连续 ID 0-3):

| ID | Class | 用途 |
|---|---|---|
| 0 | Player | 自身定位、路径起点 |
| 1 | Monster | 目标选择、攻击 |
| 2 | Platform | NavMesh 水平平台、A* 寻路 |
| 3 | Rope | 垂直导航 |

**优点**:
- 推理延迟降低 ~40% (对比多模型并发)
- 全局 NMS 彻底消除角色与怪物的识别冲突
- 训练/标注/部署三方类别一致, 无 remap 开销

HP/MP 不再由 YOLO 识别, 改由 `hp_monitor.py` 直接读取像素颜色 (更精确, 实时).

## 🧱 组件

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                  AgentV5 (orchestrator)                     │
└─────┬───────────┬───────────┬───────────┬───────────────────┘
      ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  See     │ │  Act     │ │  Heal    │ │  Think+Sense │
│ Window   │ │ Game     │ │ Auto     │ │ CombatBrain  │
│ Capture  │ │ Control- │ │ Healer + │ │   +          │
│          │ │ ler      │ │ HP       │ │   YOLO       │
│          │ │          │ │ Monitor  │ │ (Super Brain)│
└──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

### 1. See (`src/capture/`)
- `window_capture.py`: PrintWindow/DXGI 后台截屏 1600x900

### 2. Act (`src/brain/`)
- `game_controller.py`: DirectInput 后台注入 Scan Code

### 3. Heal (`src/brain/` + `src/perception/`)
- `hp_monitor.py`: 像素级 HP/MP 颜色检测 (不依赖 YOLO)
- `auto_healer.py`: HP<0.5 自动喝红, MP<0.3 自动喝蓝

### 4. Think + Sense (`src/brain/` + `src/perception/`)
- `combat_brain.py`: FSM 状态机 (scan → approach → attack → loot → patrol)
- 加载 `models/super_brain.pt`, 调用 YOLO 推理, 输出 PerceptionData

## 📡 数据流

```
WindowCapture ──frame──▶ CombatBrain
                            │
                            ▼
                       YOLO inference
                            │
                            ▼
                  PerceptionData (Player/Monster/Platform/Rope)
                            │
                            ▼
                  GlobalBus (broadcast)
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
          GameController  AutoHealer  Pathfinder
```

## 📦 训练相关

- 训练数据: `data/auto_dataset/` (5949 张, V13 0-3 标注)
- 训练脚本: `train_super_brain.py`
- 构建脚本: `scripts/build_dataset.py`
- 标注工具: `tools/web_annotator.py`
- 模型输出: `runs/detect/super_brain/weights/best.pt`
- 部署位置: `models/super_brain.pt`

详见 `.agent/EVOLUTION.md` 和 `.agent/OPERATIONS.md`.
