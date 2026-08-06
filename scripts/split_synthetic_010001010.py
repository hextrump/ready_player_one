"""
把合成数据 split 成 train/val (默认 80/20)
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "synthetic_010001010"

OUT = SRC
OUT_IMG = {"train": OUT / "images" / "train", "val": OUT / "images" / "val"}
OUT_LBL = {"train": OUT / "labels" / "train", "val": OUT / "labels" / "val"}


def main(ratio=0.8, seed=42):
    rng = random.Random(seed)
    files = sorted((SRC / "images").glob("*.jpg"))
    # 清理旧 split
    for split in ("train", "val"):
        for d in (OUT_IMG[split], OUT_LBL[split]):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    # 重置 images/ (把所有 .jpg 收回)
    all_imgs = list((SRC / "images").glob("*.jpg"))
    for img in all_imgs:
        # 如果在 images/ 根目录就保留, 子目录的移回根目录
        pass

    # 收集所有有效 pair
    pairs = []
    for img in all_imgs:
        if img.parent.name in ("train", "val"):
            continue
        lbl = SRC / "labels" / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    rng.shuffle(pairs)
    cut = int(len(pairs) * ratio)

    for split, items in [("train", pairs[:cut]), ("val", pairs[cut:])]:
        for img, lbl in items:
            shutil.copy2(img, OUT_IMG[split] / img.name)
            shutil.copy2(lbl, OUT_LBL[split] / lbl.name)
        print(f"{split}: {len(items)} 张")

    # 更新 dataset.yaml
    yaml = f"""path: data/synthetic_010001010
train: images/train
val: images/val

names:
  0: Player
  1: BlueSnail
  2: Shroom
  3: RedSnail
  4: Stump
  5: Slime
  6: OrangeMushroom
  7: GreenMushroom
  8: Platform
  9: Rope
"""
    (SRC / "dataset.yaml").write_text(yaml, encoding="utf-8")
    print(f"[OK] {SRC / 'dataset.yaml'}")


if __name__ == "__main__":
    main()