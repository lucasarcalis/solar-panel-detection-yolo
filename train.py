from ultralytics import YOLO
from roboflow import Roboflow
import os

def main():
    # 1. Télécharger le dataset depuis Roboflow
    # C'est mieux de le faire via code pour que ce soit reproductible
    rf = Roboflow(api_key="TA_CLE_API_ICI")
    project = rf.workspace("savoie").project("usmb")
    version = project.version(1) # Mets la bonne version
    dataset = version.download("yolov8")

    # 2. Configurer et Entraîner le modèle
    # On utilise le modèle de segmentation Large
    model = YOLO("yolo11l-seg.pt")

    print("🚀 Démarrage de l'entraînement...")
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0, # Mets 'cpu' si tu n'as pas de carte graphique NVIDIA sur ton PC
        project="runs/detect",
        name="solar_model"
    )
    
    print("✅ Entraînement terminé. Modèle sauvegardé dans runs/detect/solar_model")

if __name__ == "__main__":
    main()