# 🧬 Dataset Evolution Workflow (Agentic)

This workflow is used to improve the bot's accuracy in specific situations:
- **Adding a new monster** (e.g., Slime, Ribbon Pig).
- **Updating player appearance** (e.g., Change gear).
- **Improving localization** (e.g., Bounding box is skewed).

## 🚀 The 4-Stage Evolution

### Stage 1: Data Collection & Annotation
- **Tool**: `tools/live_box_annotator.py`.
- **Action**: Run the tool, use hotkeys to capture snapshots, and draw boxes (Player, Monster, HP, MP).
- **Output**: `data/entity/snapshots/` (*_raw.png + *_labels.json).

### Stage 2: Dataset Preparation
- **Tool**: `scripts/unify_dataset.py`.
- **Action**: This script merges:
  - Human-labeled snapshots (`data/entity/snapshots/`).
  - Synthetic data (`data/yolo_dataset/`).
  - Legacy YOLO sets.
- **Output**: `data/unified_dataset/` (YOLO formatted image/label splits).
- **Crucial**: Update `UNIFIED_NAMES` in `unify_dataset.py` when adding classes.

### Stage 3: Training (Super Brain)
- **Tool**: `train_super_brain.py`.
- **Action**: Trains YOLOv8 in high resolution (1280x1280).
- **Parameters**: `epochs=200`, `imgsz=1280`, `batch=4`, `patience=50`.
- **Output**: `runs/detect/super_brain_v11_highres/weights/best.pt`.

### Stage 4: Deployment & Cloud Sync
- **Tool**: `models/` directory.
- **Action**:
  1. Manually copy `runs/detect/.../best.pt` to `models/super_brain_v11.pt`.
  2. `git add models/super_brain_v11.pt`.
  3. `git commit -m "Update model: v11.3"`.
  4. `git push`.
- **Logic**: `CombatBrain` preferentially loads from `models/` for seamless multi-device deployment.

## 🛠️ Adding a New Monster
1. **Find Sprites**: Add monster PNG (with alpha) to `data/monster_db/`.
2. **Update Index**: Add entry to `data/monster_db/monster_index.json`.
3. **Synthesis**: Update `TARGET_MONSTERS` in `generate_yolo_data.py`.
4. **Re-train**: Follow Stages 2 & 3.
