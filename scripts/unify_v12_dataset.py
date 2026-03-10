import os
import json
import shutil
import cv2
import random
from pathlib import Path

# --- Configuration (V12 Unified Model) ---
BASE_PATH = r"C:\Users\hhhhhh\Documents\ready_player_one\data"
TARGET_DIR = os.path.join(BASE_PATH, "unified_dataset_v12")
VAL_RATIO = 0.15

# TARGET CLASS MAPPING (v12)
# 0: Player
# 1: Monster
# 2: HP
# 3: MP
# 4: Platform
# 5: Rope

UNIFIED_NAMES = ["Player", "Monster", "HP", "MP", "Platform", "Rope"]

# SOURCE PATHS
SOURCES = {
    "snapshots": {
        "path": os.path.join(BASE_PATH, "entity/snapshots"),
        "type": "snapshot",
        "mapping": {"Player": 0, "Monster": 1, "Wild Boar": 1, "Dark Axe Stump": 1, "Pig": 1, "Ribbon Pig": 1, "HP": 2, "MP": 3}
    },
    "entity_yolo": {
        "path": os.path.join(BASE_PATH, "yolo_entity_dataset"),
        "type": "yolo",
        "mapping": {0: 0, 1: 1, 2: 2, 3: 3}
    },
    "monster_yolo": {
        "path": os.path.join(BASE_PATH, "yolo_monster_dataset"),
        "type": "yolo",
        "mapping": {0: 1}
    },
    "terrain_yolo": {
        "path": os.path.join(BASE_PATH, "yolo_terrain_dataset"),
        "type": "yolo",
        "mapping": {0: 4, 1: 5}
    },
    "ui_yolo": {
        "path": os.path.join(BASE_PATH, "yolo_ui_dataset"),
        "type": "yolo",
        "mapping": {0: 2, 1: 3}
    },
    "synthetic_monster": {
        "path": os.path.join(BASE_PATH, "yolo_dataset"), # From generate_yolo_data.py
        "type": "yolo",
        "mapping": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1} # All monsters (1-4) map to 1
    }
}

def setup_dirs():
    if os.path.exists(TARGET_DIR):
        print(f"Cleaning existing directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(TARGET_DIR, d), exist_ok=True)

def process_snapshots(config, tag):
    source_path = config["path"]
    mapping = config["mapping"]
    print(f"Processing snapshots from {source_path}...")
    if not os.path.exists(source_path):
        print(f"  SKIPPED: Not found.")
        return
        
    json_files = [f for f in os.listdir(source_path) if f.endswith("_labels.json")]
    count = 0
    for jf in json_files:
        prefix = jf.replace("_labels.json", "")
        img_path = os.path.join(source_path, prefix + "_raw.png")
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        is_val = random.random() < VAL_RATIO
        split = "val" if is_val else "train"
        new_name = f"snap_{prefix}"
        
        # Copy image
        shutil.copy(img_path, os.path.join(TARGET_DIR, "images", split, new_name + ".png"))
        
        # Convert labels
        with open(os.path.join(source_path, jf), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        yolo_lines = []
        for box in data.get("boxes", []):
            label = box["cls"]
            if label not in mapping: continue
            cls_id = mapping[label]
            
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            bw, bh = (x2 - x1), (y2 - y1)
            cx, cy = x1 + bw/2, y1 + bh/2
            yolo_lines.append(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
            
        with open(os.path.join(TARGET_DIR, "labels", split, new_name + ".txt"), 'w') as f:
            f.write("\n".join(yolo_lines))
        count += 1
    print(f"  Done: {count} images.")

def process_yolo(config, tag):
    source_path = config["path"]
    mapping = config["mapping"]
    print(f"Processing YOLO dataset from {source_path}...")
    if not os.path.exists(source_path):
        print(f"  SKIPPED: Not found.")
        return
        
    img_dir = os.path.join(source_path, "images")
    lbl_dir = os.path.join(source_path, "labels")
    count = 0
    
    for split_in in ["train", "val"]:
        si_img = os.path.join(img_dir, split_in)
        si_lbl = os.path.join(lbl_dir, split_in)
        if not os.path.exists(si_img): continue
        
        for f in os.listdir(si_img):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            base = os.path.splitext(f)[0]
            src_img = os.path.join(si_img, f)
            src_lbl = os.path.join(si_lbl, base + ".txt")
            if not os.path.exists(src_lbl): continue
            
            is_val = random.random() < VAL_RATIO
            split_out = "val" if is_val else "train"
            new_name = f"{tag}_{base}"
            
            shutil.copy(src_img, os.path.join(TARGET_DIR, "images", split_out, new_name + os.path.splitext(f)[1]))
            
            with open(src_lbl, 'r') as fr:
                lines = fr.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                old_cls = int(parts[0])
                if old_cls in mapping:
                    new_lines.append(f"{mapping[old_cls]} {' '.join(parts[1:])}")
            
            with open(os.path.join(TARGET_DIR, "labels", split_out, new_name + ".txt"), 'w') as fw:
                fw.write("\n".join(new_lines))
            count += 1
            
    print(f"  Done: {count} images.")

def create_yaml():
    yaml_content = f"path: {Path(TARGET_DIR).as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n"
    for i, name in enumerate(UNIFIED_NAMES):
        yaml_content += f"  {i}: {name}\n"
    with open(os.path.join(TARGET_DIR, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)
    print("Created dataset.yaml")

if __name__ == "__main__":
    setup_dirs()
    for tag, config in SOURCES.items():
        if config["type"] == "snapshot":
            process_snapshots(config, tag)
        else:
            process_yolo(config, tag)
    create_yaml()
    print("\n[SUCCESS] V12 Unified Dataset Ready!")
    print(f"Path: {TARGET_DIR}")
