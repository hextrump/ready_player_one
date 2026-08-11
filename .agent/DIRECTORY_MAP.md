# 📂 Directory Map: Multi-Device Sync (Agentic)

Wait... How do I sync this project between a powerful Training PC and a mobile Laptop?

## 📁 Git-Tracked (Always Synced)
- **`src/`**: Shared brain code.
- **`models/`**: V19 单类怪模型 (`monster_v19.pt`) + 名牌模板/偏移 (`nametag/`)；旧多类模型 (`super_brain_*`) 已弃用但保留存档。
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
4. **Boot Up**: `python main.py` (Wait for `monster_v19.pt` to load).

## 🚀 Future Scalability
- **Map Addition**: 贪心移动 + 跳跃启发式已适配大部分图；复杂多层图靠跳发补刀/登台跳/脱困跳兜底，无需每图建导航。
- **UI Expansion**: 如需 UI 元素检测，可给 `monster_v19.pt` 重训加入类，或新增专门小模型。
- **Bot Behavior**: 针对不同职业的战斗逻辑可做 `CombatBrain` 子类或策略参数化。
