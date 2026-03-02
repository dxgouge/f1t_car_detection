from ultralytics import YOLO
import os
def create_dataset_config():
   


    # Write dataset.yaml file
    #yaml_path = os.path.join(os.path.dirname(__file__), 'f1t_yolo_dataset.yaml')
    #with open(yaml_path, 'w') as f:
    #   f.write(dataset_config)

    # !!Your path to the dataset.yaml file!!
    yaml_path = './f1tenth.v2i.yolov11/data.yaml'
    print(f"Dataset configuration created at: {yaml_path}")
    return yaml_path

def train_model(yaml_path):
    """Train the YOLO model for car detection."""
    # Load a pre-trained YOLO model for object detection
    print("Loading YOLO model...")

    # Could be better model options?
    model = YOLO('yolo11n.pt')  #- loads existing pre-trained model for training
    print("Model loaded successfully!")

    # Train the model
    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=30,  # Number of training epochs
        imgsz=640,  # Image size for detection
        batch=4,    # Batch size
        device='cpu',  # Use GPU for faster training - Change to "cuda" on windows/lin, "mps" for mac - "cpu" if ur gpu blows
        project=os.path.join(os.getcwd(), 'f1t_yolo_training'),
        name='f1t_car_detection',
        val=True,   # Enable validation
        patience=10,  # Early stopping patience
        save_period=10,  # Save checkpoint every 10 epochs
        workers=2,   # Reduced workers for stability
        # Data augmentation for robustness
        hsv_h=0.015,  # Hue variation
        hsv_s=0.7,    # Saturation variation
        hsv_v=0.4,    # Value/brightness variation
        degrees=10,   # Rotation variation
        translate=0.1, # Translation variation
        scale=0.5,    # Scale variation
        shear=2.0,    # Shear variation
        perspective=0.0, # Perspective variation
        flipud=0.0,   # Vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
        copy_paste=0.1, # Copy-paste augmentation
        auto_augment='randaugment'  # Auto augmentation
    )

    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    return results
def test_model(results):
    """Test the trained model on a sample image."""
    # Test the trained model
    print("\nTesting the trained model...")
    trained_model = YOLO(f'{results.save_dir}/weights/best.pt')

    # Test on a sample image
    # !!Replace this path with the actual path to your test image!!
    test_image = './f1tenth.v2i.yolov11/test/images/4_jpg.rf.63a6ef47810a41733934a5d485d1fb0a.jpg'
    if os.path.exists(test_image):
        test_results = trained_model.predict(source=test_image, show=True, save=True)
        print("Test completed!")
    else:
        print(f"Test image not found: {test_image}")
def main():
    """Main function to train and test the YOLO model."""
    # Create dataset configuration
    yaml_path = create_dataset_config()
    
    # Train the model
    results = train_model(yaml_path)
    
    # Test the model
    test_model(results)
if __name__ == '__main__':
    main()