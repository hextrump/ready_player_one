# 🎮 Bot Operations & Debugging Guide

This guide is for starting the bot, monitoring its health, and troubleshooting common issues.

## 🏁 How to Start
```powershell
# 1. Activate Environment
conda activate py313

# 2. Run Main Bot
python main.py
```

## 📊 Monitoring Health
- **Live Logs**: Check `logs/combat_bot.log` for logic errors (e.g., "Monster out of range", "HP low").
- **Visual Check**:
  - The bot displays its perception window by default.
  - Green boxes: Player.
  - Red/Purple boxes: Monsters.
  - Blue/White bars: HP/MP detection.

## 🛠️ Common Issues & Fixes

### Issue: Player misidentified as a Monster
- **Cause**: Bounding boxes overlap too much, or player sprites lack diversity.
- **Fix**:
  1. Capture 20+ snapshots with `live_box_annotator.py`.
  2. Label the player tightly.
  3. Re-run `scripts/unify_dataset.py` and `train_super_brain.py`.
- **Logic Safeguard**: `CombatBrain.find_targets` filters out any monster box that covers >70% of the player's confirmed location.

### Issue: Bot is "lagging" or FPS is low
- **Cause**: High inference resolution (1280x1280) is GPU-heavy.
- **Fix**:
  - Reduce `imgsz` in `CombatBrain` (e.g., to 640), though accuracy will drop.
  - Ensure `torch.cuda.is_available()` is true.
  - Use `models/super_brain_v11.pt` as it is optimized.

### Issue: Bot doesn't move or use skills
- **Cause**: Global key capture issues or `ActionTranslator` coordinate mismatch.
- **Fix**:
  - Ensure game window is focused and has administrative privileges.
  - Check `config.yaml` for key mappings.

## 🧪 Testing
- **Perception Test**: `pytest tests/test_perception.py` (Tests if model can see a sample image).
- **Control Test**: `python tests/test_fsm.py` (Tests if the state machine logic transitions correctly).
