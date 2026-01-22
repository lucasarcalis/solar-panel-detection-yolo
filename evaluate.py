import matplotlib.pyplot as plt

def plot_model_metrics(metrics_dict):
    """
    Plots the validation metrics (Box vs Mask) for the YOLO model.
    """
    
    # Mapping keys to readable labels
    labels_map = {
        "metrics/precision(B)": "Precision (Box)",
        "metrics/recall(B)": "Recall (Box)",
        "metrics/mAP50(B)": "mAP50 (Box)",
        "metrics/mAP50-95(B)": "mAP50-95 (Box)",
        "metrics/precision(M)": "Precision (Mask)",
        "metrics/recall(M)": "Recall (Mask)",
        "metrics/mAP50(M)": "mAP50 (Mask)",
        "metrics/mAP50-95(M)": "mAP50-95 (Mask)",
    }

    # Filter and extract data
    keys = [k for k in metrics_dict.keys() if k in labels_map]
    values = [metrics_dict[k] for k in keys]
    labels = [labels_map.get(k, k) for k in keys]

    # Plot configuration
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='steelblue')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("YOLOv11 Validation Metrics")
    plt.tight_layout()
    
    # Save the plot locally
    plt.savefig("validation_metrics.png")
    print("Metrics plot saved as validation_metrics.png")
    plt.show()

# Example usage (uncomment when running with a trained model)
# from ultralytics import YOLO
# model = YOLO("path/to/best.pt")
# metrics = model.val()
# plot_model_metrics(metrics.results_dict)