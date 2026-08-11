---
description: Retrain the V19 single-class Monster model with current data
---

该工作流自动化"整理怪数据集 → 训练 V19 单类怪模型 → 部署"的流程。

1. 收集快照: 运行 bot 时 `data_collector` 自动心跳截图到 `data/auto_dataset/`;
   也可主动采集误检样本。
2. 整理数据集: 把快照整理/标注为 `data/yolo_monster_dataset/dataset.yaml` (单类 Monster)。
3. 启动训练:
// turbo
`python archive/train_monster_v19.py`

4. 训练完成后, 找到 `runs/detect/monster_v19_pig/weights/best.pt`。

5. 复制到 models:
// turbo
`cp runs/detect/monster_v19_pig/weights/best.pt models/monster_v19.pt`

6. 提交并推送:
// turbo
`git add models/monster_v19.pt; git commit -m "update: monster v19 weights"; git push origin main`
