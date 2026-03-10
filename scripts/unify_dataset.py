import os
import json
import shutil
import cv2
import random
from pathlib import Path

# --- Configuration ---
SOURCE_SNAPSHOTS = r"C:\Users\hhhhhh\Documents\ready_player_one\data\entity\snapshots"
SOURCE_YOLO_MONSTER = r"C:\Users\hhhhhh\Documents\ready_player_one\data\yolo_dataset"
SOURCE_YOLO_ENTITY = r"C:\Users\hhhhhh\Documents\ready_player_one\data\yolo_entity_dataset"

TARGET_DIR = r"C:\Users\hhhhhh\Documents\ready_player_one\data\unified_dataset"
VAL_RATIO = 0.15

UNIFIED_NAMES = [
    "Player",           # 0
    "HP",               # 1
    "MP",               # 2
    "Monster",          # 3 (General)
    "Wild Boar",        # 4
    "Dark Axe Stump",   # 5
    "Pig",              # 6
    "Ribbon Pig"        # 7
]

# Mapping from source datasets to unified class IDs
# generate_yolo_data [0: Player, 1: Wild Boar, 2: Dark Axe Stump, 3: Pig, 4: Ribbon Pig]
MAP_MONSTER = {0: 0, 1: 4, 2: 5, 3: 6, 4: 7} 
MAP_ENTITY = {0: 0, 1: 3, 2: 1, 3: 2} # 0=Player->0, 1=Monster->3, 2=HP->1, 3=MP->2
MAP_SNAPSHOTS = {"Player": 0, "Monster": 3, "HP": 1, "MP": 2}

def setup_dirs():
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(TARGET_DIR, d), exist_ok=True)

def process_snapshots():
    print("Processing snapshots...")
    if not os.path.exists(SOURCE_SNAPSHOTS):
        print(f"Skipping {SOURCE_SNAPSHOTS} (not found)")
        return
    json_files = [f for f in os.listdir(SOURCE_SNAPSHOTS) if f.endswith("_labels.json")]
    count = 0
    for jf in json_files:
        prefix = jf.replace("_labels.json", "")
        img_path = os.path.join(SOURCE_SNAPSHOTS, prefix + "_raw.png")
        json_path = os.path.join(SOURCE_SNAPSHOTS, jf)
        
        if not os.path.exists(img_path): continue
        
        # Determine split
        is_val = random.random() < VAL_RATIO
        split = "val" if is_val else "train"
        
        # Load image size
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # Copy image
        target_img_name = f"snap_{prefix}.png"
        shutil.copy(img_path, os.path.join(TARGET_DIR, "images", split, target_img_name))
        
        # Convert labels
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        yolo_lines = []
        for box in data.get("boxes", []):
            cls_str = box["cls"]
            if cls_str not in MAP_SNAPSHOTS: continue
            cls_id = MAP_SNAPSHOTS[cls_str]
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            bw = (x2 - x1)
            bh = (y2 - y1)
            cx = x1 + bw / 2
            cy = y1 + bh / 2
            
            yolo_lines.append(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
            
        with open(os.path.join(TARGET_DIR, "labels", split, f"snap_{prefix}.txt"), 'w') as f:
            f.write("\n".join(yolo_lines))
        count += 1
    print(f"  Done. Processed {count} snapshots.")

def process_yolo_dataset(source_path, mapping, tag):
    print(f"Processing {tag} dataset from {source_path}...")
    if not os.path.exists(source_path):
        print(f"Skipping {source_path} (not found)")
        return
    img_dir = os.path.join(source_path, "images")
    lbl_dir = os.path.join(source_path, "labels")
    
    count = 0
    # YOLO datasets usually have train/val subdirs
    for split_in in ["train", "val"]:
        src_img_split = os.path.join(img_dir, split_in)
        src_lbl_split = os.path.join(lbl_dir, split_in)
        
        if not os.path.exists(src_img_split): continue
        
        files = [f for f in os.listdir(src_img_split) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            name = os.path.splitext(f)[0]
            src_img = os.path.join(src_img_split, f)
            src_lbl = os.path.join(src_lbl_split, name + ".txt")
            
            if not os.path.exists(src_lbl): continue
            
            # Resplit randomly for the unified dataset
            is_val = random.random() < VAL_RATIO
            split_out = "val" if is_val else "train"
            
            # Copy image
            new_name = f"{tag}_{name}"
            shutil.copy(src_img, os.path.join(TARGET_DIR, "images", split_out, new_name + os.path.splitext(f)[1]))
            
            # Remap labels
            with open(src_lbl, 'r') as fr:
                lines = fr.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                old_cls = int(parts[0])
                if old_cls in mapping:
                    new_cls = mapping[old_cls]
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
            
            with open(os.path.join(TARGET_DIR, "labels", split_out, new_name + ".txt"), 'w') as fw:
                fw.write("\n".join(new_lines))
            count += 1
            
    print(f"  Done. Processed {count} files from {tag}.")

def create_yaml():
    yaml_content = f"""path: {Path(TARGET_DIR).as_posix()}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(UNIFIED_NAMES):
        yaml_content += f"  {i}: {name}\n"
        
    with open(os.path.join(TARGET_DIR, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)
    print("Created dataset.yaml")

if __name__ == "__main__":
    setup_dirs()
    process_snapshots()
    process_yolo_dataset(SOURCE_YOLO_MONSTER, MAP_MONSTER, "monster")
    process_yolo_dataset(SOURCE_YOLO_ENTITY, MAP_ENTITY, "entity")
    create_yaml()
    print("\nDataset unification complete!")
    print(f"Target: {TARGET_DIR}")
