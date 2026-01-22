from ultralytics import YOLO
import cv2
import sys

# Charger le modèle entraîné (assure-toi de copier ton best.pt ici)
model_path = "models/best.pt" 

def detect(image_path):
    model = YOLO(model_path)
    
    # Faire l'inférence
    results = model.predict(source=image_path, save=True, conf=0.4)
    
    # Afficher le résultat (optionnel)
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        cv2.imshow("Resultat", im_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "inference_images/test.jpg" # Image par défaut
        
    detect(img_path)