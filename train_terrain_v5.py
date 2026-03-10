from ultralytics import YOLO

if __name__ == '__main__':
    # Continue training terrain using pretrained 
    model = YOLO('runs/detect/runs/detect/terrain_v4_ludibrium/weights/best.pt') 

    # Train model on the expanded dataset combining old ones and new pig beach ones
    results = model.train(
        data='data/yolo_terrain_dataset/dataset.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        project='runs/detect',
        name='terrain_v5_pig_beach',
        device=0,
        workers=4,
        verbose=True,
    )
