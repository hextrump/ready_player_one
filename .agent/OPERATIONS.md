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
- **Live Logs**: Check `logs/agent.log` for logic errors (e.g., "ATTACK", "脱困跳", "PATROL").
- **Visual Check**:
  - The bot displays its perception window by default.
  - Red cross: Player position (from nametag).
  - Orange boxes: Monsters (V19 detections).
  - Green box: Nametag match.
  - Red/Blue bars: HP/MP detection.

## 🛠️ Common Issues & Fixes

### Issue: Player misidentified as a Monster
- **Cause**: V19 会把玩家自己误检成 Monster。
- **Fix**:
  1. 采集名牌模板/偏移 (`tools/capture_nametag.py`), 确保名牌定位可用。
  2. 收集误检样本, 重训 `monster_v19.pt` (见 `EVOLUTION.md`)。
- **Logic Safeguard**: `CombatBrain.find_targets` 过滤掉与玩家已确认位置重叠 >30% 的怪框。

### Issue: Bot is "lagging" or FPS is low
- **Cause**: YOLO 推理或抓帧占用 GPU/CPU。
- **Fix**:
  - 主模型 `imgsz=640` 已够 (V19 单类怪, 不需要 1280 超清)。
  - Ensure `torch.cuda.is_available()` is true.
  - 后台视觉线程 ~7fps; 若仍卡, 检查抓帧是否被其他窗口遮挡。

### Issue: Bot doesn't move or use skills
- **Cause**: 后台键盘注入失败或贪心移动卡死。
- **Fix**:
  - Ensure game window is visible (not minimized) and script runs with administrative privileges.
  - 卡在台阶时观察"脱困跳"日志是否触发; 若多层图频繁卡住, 设 `FLAT_MODE=True` 或换平层图。
  - 按键绑定在 `src/brain/game_controller.py` (SCAN/VK 表); `config.yaml` 仅文档性配置。

## 🧪 Testing
- **冒烟测试**: `python -c "from src.brain.combat_brain import CombatBrain; b = CombatBrain()"` — 确认 V19 加载、无 import 报错。
- **实机验证**: `python main.py` (或 `--process <exe>`) → F1 开打, 观察"打怪/靠近/巡逻"三态 + 名牌校准 HUD。
- 注: `tests/` 下现有测试是旧 event-driven 架构的遗留, 引用已不存在的模块 (yolo_detector 等), 暂不可用。
