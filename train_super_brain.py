"""
训练 Super Brain V13 (统一 4 类: Player/Monster/Platform/Rope)

前置: 先运行 python scripts/build_dataset.py 构建 data/super_brain_train/

输出: runs/detect/super_brain/weights/best.pt
部署: 复制 best.pt 到 models/super_brain.pt
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "super_brain_train" / "dataset.yaml"


def main():
    if not DATA_YAML.exists():
        raise SystemExit(
            f"未找到 {DATA_YAML}\n请先运行: python scripts/build_dataset.py"
        )

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=300,
        imgsz=960,
        batch=4,
        project="runs/detect",
        name="super_brain",
        device=0,
        workers=4,
        patience=50,
        verbose=True,
    )


if __name__ == "__main__":
    main()
