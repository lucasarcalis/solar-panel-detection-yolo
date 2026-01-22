import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # Setup Roboflow authentication
    # Ideally, keep the API key private or use an environment variable
    api_key = "PUT_YOUR_API_KEY_HERE" 
    rf = Roboflow(api_key=api_key)

    # Access the USMB project
    project = rf.workspace("savoie").project("usmb")
    
    # Download the dataset (Version 2 as used in the notebook)
    # Using yolov8 format which is compatible with YOLOv11
    version = project.version(2)
    dataset = version.download("yolov8")

    print(f"Dataset downloaded to: {dataset.location}")

    # Load the YOLOv11 Large Segmentation model
    model = YOLO("yolo11l-seg.pt")

    # Start training
    # Parameters adapted for Colab GPU usage
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        project="savoie_results",
        name="solar_segmentation_v1"
    )

if __name__ == "__main__":
    main()
