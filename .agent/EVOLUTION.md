# 🧬 Dataset Evolution Workflow (Agentic)

该工作流用于提升战斗模型在特定场景下的识别精度:
- **加入新怪** (e.g., Slime, Ribbon Pig)。
- **修正误检** (e.g., 把玩家误检成 Monster, 框偏斜)。
- **玩家定位兜底** (名牌模板失效时)。

## 当前架构: 单类 Monster 模型 (V19) + 名牌定位玩家

- `models/monster_v19.pt` 只检测 `Monster` 一个类 (`{0: Monster}`)。
- 玩家位置由 `src/perception/nametag_locator.py` 模板匹配玩家静态名牌锚定,
  通过 `tools/capture_nametag.py` 采集名牌模板 + 偏移 (`models/nametag/`)。
- 训练脚本: `archive/train_monster_v19.py` (yolov8n, imgsz=640, epochs=150)。

## 🚀 3-Stage Evolution (Monster Model)

### Stage 1: 数据采集与标注
- **自动采集**: `CombatBrain` 后台心跳截图 (`src/brain/data_collector.py`) 定期存
  带 YOLO 标签的快照到 `data/auto_dataset/`。
- **主动标注**: 用标注工具对快照打怪框; 关注误检样本 (玩家被当怪)。
- **数据集**: 整理为 `data/yolo_monster_dataset/dataset.yaml` (单类 Monster)。

### Stage 2: 训练 (V19+)
- **Tool**: `python archive/train_monster_v19.py`
- **Parameters**: `epochs=150`, `imgsz=640`, `batch=16`, 基座 `yolov8n.pt`。
- **产出**: `runs/detect/monster_v19_pig/weights/best.pt` → 复制为 `models/monster_v19.pt`。

### Stage 3: 部署与云同步
- 把 `models/monster_v19.pt` 提交推送到远端 (Syncthing/git 均可)。

## 🧱 Class Mapping (当前运行时)
| 来源 | 检测目标 | 用途 |
|---|---|---|
| `monster_v19.pt` | Monster | 选目标, 战斗 |
| `nametag_locator.py` | 玩家名牌 | 玩家坐标锚定 |
| `hp_monitor.py` | HP/MP 条 | 自动喝药 |

## 🛠️ 加入新怪
1. **找素材**: 把怪 PNG 放入 `data/monster_db/`。
2. **合成样本**: 更新生成脚本的 `TARGET_MONSTERS`, 生成训练图。
3. **重训**: 走 Stage 1-2, 产出新的 `monster_v19.pt`。
