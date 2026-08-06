# 📂 Directory Map

## 📁 Git-Tracked (Always Synced)
- **`src/`**: Bot 业务代码 (window capture, brain, perception, navigation, image_utils)
- **`scripts/`**: 数据集构建 (`build_dataset.py`), 迁移 (`migrate_labels.py`, `resize_dataset.py`)
- **`tools/`**: 标注 & 可视化 (`web_annotator.py`, `check_auto_dataset.py`, `web_annotator.html`)
- **`models/`**: 训练好的权重 (云同步或 LFS)
- **`.agent/`**: 文档与演进指引
- **`plan.md`**: 项目路线图
- **`train_super_brain.py`**: 训练入口
- **`main.py`**: Bot 启动入口
- **`generate_yolo_data.py`**: 合成数据生成器 (可选)

## 📁 Git-Ignored (本地管理)
- **`data/auto_dataset/`**: 唯一原始数据源 (1366×768 jpg + V13 0-3 labels)
- **`data/super_brain_train/`**: 训练集派生 (运行 `build_dataset.py` 自动生成)
- **`runs/`**: YOLO 训练输出
- **`train.log`**: 训练日志
- **`logs/`**: 运行日志
- **`.venv/`**: Python 环境

## 📐 规范画布

**1366×768** 是统一画布, 训练集、运行时抓帧、标注器都按这个尺寸工作.
详见 `src/utils/image_utils.py` 的 `CANONICAL_SIZE`.

## 🔄 新机器部署流程

1. `git pull origin main`
2. `pip install -r requirements.txt`
3. 把 `data/auto_dataset/` (4.6GB) 通过 USB/网盘同步过来
4. `python scripts/build_dataset.py` 构建训练集
5. `python train_super_brain.py` 训练 (或直接用已同步的 `models/super_brain.pt`)
6. `python main.py` 启动 bot

## 🏷️ V13 类别速查

| ID | Class | HTML | BGR |
|---|---|---|---|
| 0 | Player | `#0066cc` | (255, 0, 0) |
| 1 | Monster | `#cc0000` | (0, 0, 255) |
| 2 | Platform | `#00aa00` | (0, 255, 0) |
| 3 | Rope | `#00aaaa` | (255, 255, 0) |
