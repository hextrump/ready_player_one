# 📂 Directory Map: Multi-Device Sync (Agentic)

Wait... How do I sync this project between a powerful Training PC and a mobile Laptop?

## 📁 Git-Tracked (Always Synced)
- **`src/`**: Shared brain code.
- **`models/`**: Unified weight files (Super Brain).
- **`.agent/`**: The documentation and evolution instructions (this folder!).
- **`config.yaml`**: The shared configuration.
- **`plan.md`**: Roadmap and progress.

## 📁 Git-Ignored (Not Synced, Locally Managed)
- **`data/`**: Large raw datasets (5.8GB+).
- **`runs/`**: Intermediate training logs/weights.
- **`logs/`**: Local runtime logs.
- **`archive/`**: Obsolete/Experiment scripts.

## 🔄 The Sync Workflow
To move the bot to a new computer (Laptop):
1. **Pull Code**: `git pull origin main`.
2. **Setup Dependencies**: `pip install -r requirements.txt`.
3. **Move Data (Optional)**: Move `data/monster_db/` (sprites) and `data/entity/snapshots/` (samples) via a USB/Network drive to enable local training.
4. **Boot Up**: `python main.py` (Wait for `super_brain_v11.pt` to load).

## 🚀 Future Scalability
- **Map Addition**: Create new subfolders in `src/navigation/` for each map (Henesys, Kerning).
- **UI Expansion**: Add UI elements (Inventory, Minimap) to the `UNIFIED_NAMES` for Super Brain detection.
- **Bot Behavior**: Add specialized FSM subclasses in `src/brain/` for different classes (Mage, Bowman).
