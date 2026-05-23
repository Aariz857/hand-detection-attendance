from ultralytics import YOLO

def train_model():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train the model on your dataset
    results = model.train(
        data='data.yaml',
        epochs=100,        # Increased epochs for better accuracy
        imgsz=640,
        patience=20,       # Stop early if no improvement
        batch=16,
        device='cpu'       # Change to '0' for GPU
    )
    print("Training complete. Model saved in runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_model()
