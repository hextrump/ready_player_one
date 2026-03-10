from ultralytics import YOLO

if __name__ == '__main__':
    # Continue training terrain using the previous Pig Beach model
    model = YOLO('runs/detect/runs/detect/terrain_v5_pig_beach/weights/best.pt') 

    # Train model on the expanded dataset combining old ones and new Henesys East ones
    results = model.train(
        data='data/yolo_terrain_dataset/dataset.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        project='runs/detect',
        name='terrain_v6_henesys',
        device=0,
        workers=4,
        verbose=True,
    )
