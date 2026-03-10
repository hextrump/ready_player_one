from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 我们使用 yolov8n 因为它推理速度最快，适合实时游戏
    model = YOLO('yolov8n.pt') 

    # 开始训练通用实体大模型 (Super Brain V5)
    # 包含: Player, HP, MP, Monster (General), Wild Boar, Dark Axe Stump
    results = model.train(
        data='C:/Users/hhhhhh/Documents/ready_player_one/data/unified_dataset/dataset.yaml',
        epochs=200,             # 更精细的训练轮数
        imgsz=1280,             # 匹配 inference 高清分辨率，解决定位“歪”的问题
        batch=4,               # 1280x1280 在 2080S 上可能需要小 batch 否则 OOM
        project='runs/detect',
        name='super_brain_v11_highres', # V11 高清版
        device=0,
        workers=4,
        patience=50,
        verbose=True,
    )
