# 🧬 Dataset Evolution Workflow (Agentic)

This workflow is used to improve the bot's accuracy in specific situations:
- **Adding a new monster** (e.g., Slime, Ribbon Pig).
- **Updating player appearance** (e.g., Change gear).
- **Improving localization** (e.g., Bounding box is skewed).

## 🚀 The 5-Stage Evolution (Unified Strategy)

### Stage 1: Data Collection & Annotation
- **Tool**: `tools/live_box_annotator.py`.
- **Labels**: `Player`, `Monster`, `HP`, `MP`, `Platform`, `Rope`.
- **Target**: Ensure every snapshot includes terrain data.

### Stage 2: Unified Dataset Preparation
- **Tool**: `scripts/unify_dataset.py`.
- **Action**: Merges ALL data sources into a single class mapping:
  - `0`: Player
  - `1`: Monster
  - `2`: HP
  - `3`: MP
  - `4`: Platform
  - `5`: Rope
- **Output**: `data/unified_dataset_v12/`.

### Stage 3: Training (Super Brain V12+)
- **Tool**: `train_super_brain.py`.
- **Parameters**: `epochs=300`, `imgsz=1280`, `batch=4`.
- **Goal**: High-precision detection of both moving entities and static terrain.

### Stage 4: Evaluation & Calibration
- **Action**: Test detection in complex maps with many "gaps".
- **Metric**: Zero synchronization error between Entity position and Platform position.

### Stage 5: Deployment & Cloud Sync
- **Tool**: `models/super_brain_v12.pt`.
- **Auto-Sync**: Git add, commit, and push.

## 🧱 Unified Class Mapping Table
| ID | Class Name | Usage in Code |
|---|---|---|
| 0 | Player | Localization, Path start point |
| 1 | Monster | Target selection, Combat |
| 2 | HP | Perception data -> AutoHealer |
| 3 | MP | Perception data -> AutoHealer |
| 4 | Platform | NavMesh construction, A* pathing |
| 5 | Rope | Vertical navigation |

## 🛠️ Adding a New Monster
1. **Find Sprites**: Add monster PNG (with alpha) to `data/monster_db/`.
2. **Update Index**: Add entry to `data/monster_db/monster_index.json`.
3. **Synthesis**: Update `TARGET_MONSTERS` in `generate_yolo_data.py`.
4. **Re-train**: Follow Stages 2 & 3.
