from ultralytics import YOLO

if __name__ == '__main__':
    # Create a new YOLO model from scratch or load a pretrained one
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model with the new entity dataset (Slime City)
    results = model.train(
        data='data/yolo_entity_dataset/dataset.yaml',  # point to our new yaml
        epochs=100,             # More epochs for better results
        imgsz=640,
        batch=16,
        project='runs/detect',
        name='entity_v18_pig_beach',  
        device=0,              # Use GPU
        workers=4,
        verbose=True,
    )
