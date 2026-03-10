from ultralytics import YOLO

if __name__ == '__main__':
    # Start with a pretrained model for better transfer learning
    model = YOLO('yolov8n.pt')

    # Train the model with the expanded monster ONLY dataset (now including pigs)
    results = model.train(
        data='data/yolo_monster_dataset/dataset.yaml',
        epochs=150,  # Slightly more epochs since dataset has more varied items now
        imgsz=640,
        batch=16,
        project='runs/detect',
        name='monster_v19_pig',      # V19 for pigs
        device=0,
        workers=4,
        verbose=True,
    )
