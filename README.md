# Solar Panel Detection & Segmentation (USMB)

This project focuses on detecting and segmenting photovoltaic panels on rooftops using aerial imagery. It was developed as part of a project at Université Savoie Mont Blanc.

We use the **YOLOv11-Large** model for instance segmentation to accurately estimate the surface area of the panels.

## Project Structure

* `scripts/`: Python scripts for training and evaluation.
* `notebooks/`: Jupyter notebooks used for initial experiments on Google Colab.
* `requirements.txt`: List of dependencies.

## Dataset

The dataset is managed via **Roboflow**. It contains annotated images of rooftops with solar panels.
The script automatically downloads the latest version of the dataset using the Roboflow API.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
