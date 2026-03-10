---
description: Automatically retrain the Super Brain model with current data
---

This workflow automates the process of unifying datasets and launching the training for the Super Brain (YOLOv8).

1. Ensure snapshot labels are up to date.
2. Run dataset unification:
// turbo
`python scripts/unify_dataset.py`

3. Generate synthetic background data (Optional but recommended):
// turbo
`python generate_yolo_data.py`

4. Launch training:
// turbo
`python train_super_brain.py`

5. After training completes, look for the `best.pt` in `runs/detect/super_brain_v11_highres/weights/`.

6. Copy to models:
// turbo
`cp runs/detect/super_brain_v11_highres/weights/best.pt models/super_brain_v11.pt`

7. Commit and Push:
// turbo
`git add models/super_brain_v11.pt; git commit -m "update: super brain model weights"; git push origin main`
