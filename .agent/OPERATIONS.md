# 🎮 Bot Operations & Debugging Guide

## 📐 规范画布: 1366×768

训练集、运行时抓帧、标注器全部按 1366×768 工作.
游戏窗口分辨率不重要 — `WindowCapture` 会自动 letterbox 到 1366×768.

## 🏁 启动

```powershell
conda activate py313    # 或激活项目 venv
python main.py
```

## ⌨️ 运行时热键

| 按键 | 作用 |
|---|---|
| F1 | 启用自动挂机 |
| F | 停止挂机 (standby) |
| Ctrl+C | 完全退出 |

## 📊 监控

- **运行日志**: `logs/combat_bot.log`
- **可视化窗口**: bot 默认弹出 perception 窗口
  - 🟦 蓝框 = Player (0)
  - 🟥 红框 = Monster (1)
  - 🟩 绿框 = Platform (2)
  - 🟨 青框 = Rope (3)

## 🔄 训练 / 再训练

```powershell
# 1. 标注 (可选)
python tools/web_annotator.py

# 2. 构建训练集
python scripts/build_dataset.py

# 3. 训练
python train_super_brain.py

# 4. 部署
cp runs/detect/super_brain/weights/best.pt models/super_brain.pt
```

详见 `.agent/EVOLUTION.md`.

## 🔄 换游戏分辨率

```python
# 1. 改 src/utils/image_utils.py 的 CANONICAL_SIZE
# 2. 改 src/capture/window_capture.py 的 CANONICAL_SIZE (与 1 保持一致)

# 3. 重做训练集 (letterbox 所有图到新尺寸)
python scripts/resize_dataset.py --backup

# 4. 重建 + 重训
python scripts/build_dataset.py
python train_super_brain.py
```
labels 不需要改 (归一化坐标).

## 🛠️ 常见问题

### Player 被误识别为 Monster
- `tools/check_auto_dataset.py --limit 30` 看标注
- `tools/web_annotator.py` 用 Delete 模式微调 Player 框
- 重跑 build + train

### Bot 延迟高 / FPS 低
- `train_super_brain.py` 里降 `imgsz` (e.g. 640)
- 确认 `torch.cuda.is_available()` 为 true

### Bot 不动 / 不放技能
- 游戏窗口需要管理员权限
- 检查 `config.yaml` 键位映射

## 🧪 测试

```powershell
pytest tests/test_perception.py
python tests/test_fsm.py
```

## 🗺️ meowdb 数据 (B 方案)

```powershell
# 下载 minimap 库 + 索引 (一次性)
.venv/Scripts/python.exe scripts/build_map_db.py --all

# 手工标 map ID
.venv/Scripts/python.exe tools/web_map_id.py    # http://localhost:8081

# 比对 (待做) YOLO vs meowdb
.venv/Scripts/python.exe scripts/compare_vision.py <image>
```
