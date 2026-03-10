from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 核心：Super Brain V12 -- 大统一模型 (Unified Perception)
    # 包含: 0:Player, 1:Monster, 2:HP, 3:MP, 4:Platform, 5:Rope
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data='C:/Users/hhhhhh/Documents/ready_player_one/data/unified_dataset_v12/dataset.yaml',
        epochs=300,             # V12 复杂度更高，增加轮数
        imgsz=960,              # 960 是性能与精度的甜点位，训练出的模型在 640 下依然很强
        batch=4,               
        project='runs/detect',
        name='super_brain_v12_unified', 
        device=0,
        workers=4,
        patience=50,
        verbose=True,
    )
