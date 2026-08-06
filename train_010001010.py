"""
训练 010001010 专属模型 (10 类: Player + 7 怪物 + Platform + Rope)

数据来源:  scripts/gen_yolo_010001010.py 生成的合成数据
          data/synthetic_010001010/

输出: runs/detect/super_brain_010001010/weights/best.pt
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "synthetic_010001010" / "dataset.yaml"


def main():
    if not DATA_YAML.exists():
        raise SystemExit(
            f"未找到 {DATA_YAML}\n请先运行: python scripts/gen_yolo_010001010.py --count 300"
        )
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=200,
        imgsz=960,
        batch=8,
        project="runs/detect",
        name="super_brain_010001010",
        device=0,
        workers=2,
        patience=30,
        verbose=True,
    )


if __name__ == "__main__":
    main()