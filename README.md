# ☀️ Solar Panel Segmentation (YOLOv11)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v11-green)
![Roboflow](https://img.shields.io/badge/Data-Roboflow-purple)

This project leverages State-of-the-Art (SOTA) computer vision to automatically detect and segment photovoltaic panels from aerial imagery (satellite or drone).

Unlike standard object detection (bounding boxes), this project uses **Instance Segmentation** to outline the exact shape of the panels, allowing for precise surface area estimation ($m^2$) and better handling of inclined roofs.

## 🖼️ Demo
*(Place a screenshot of your results here. E.g., an image showing the colored masks on a roof)*
![Result Example](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

## 🚀 Key Features
- **Instance Segmentation**: Pixel-perfect masking of solar panels using YOLOv11-Large-Seg.
- **Automated Pipeline**: Dataset management and preprocessing via Roboflow API.
- **High Precision**: Optimized for aerial views (USMB/Savoie dataset).
- **Scalable**: Ready for training on GPU (Google Colab T4/A100).

## 🛠️ Installation

1. **Clone the repository**
   ```bash
