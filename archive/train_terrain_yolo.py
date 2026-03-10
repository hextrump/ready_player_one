from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 使用基础模型开始训练地形识别 (废弃都市版)
    model = YOLO('yolov8n.pt') 

    # 训练模型
    results = model.train(
        data='data/yolo_terrain_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        project='runs/detect',
        name='terrain_v4_ludibrium',
        device=0,
        workers=4,
        verbose=True,
    )
