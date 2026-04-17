import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

projectRoot = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(projectRoot))

from src.models.train import prepare_dataloaders
from src.models.cnn_estimator import BatteryHealthCNN, get_compute_device

def evaluate_and_plot():
    device = get_compute_device()
    processedCsvPath = projectRoot / "data" / "processed" / "B0005_discharge.csv"
    modelSavePath = projectRoot / "src" / "models" / "battery_cnn_weights.pth"
    
    if not modelSavePath.exists():
        raise FileNotFoundError("Model weights not found. Run train.py first.")

    # We only need the test loader for inference
    _, testLoader = prepare_dataloaders(processedCsvPath, batchSize=1)
    
    model = BatteryHealthCNN().to(device)
    model.load_state_dict(torch.load(modelSavePath, map_location=device))
    model.eval()
    
    actual_capacities = []
    predicted_capacities = []
    
    print("Running inference on test dataset...")
    
    with torch.no_grad():
        for batchX, batchY in testLoader:
            batchX = batchX.to(device)
            prediction = model(batchX)
            
            # Move data back to CPU for matplotlib
            predicted_capacities.append(prediction.cpu().item())
            actual_capacities.append(batchY.item())

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(actual_capacities, label='Actual Capacity (Ah)', color='black', linewidth=2)
    plt.plot(predicted_capacities, label='Predicted Capacity (Ah)', color='red', linestyle='--', linewidth=1.5)
    
    plt.title('Battery Capacity Degradation: Actual vs Predicted')
    plt.xlabel('Test Sequence Timeline (Cycles)')
    plt.ylabel('Capacity (Ampere-hours)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save the plot to disk instead of opening a window
    plotPath = projectRoot / "data" / "degradation_plot.png"
    plt.savefig(plotPath, dpi=300, bbox_inches='tight')
    print(f"Evaluation complete. Plot saved to: {plotPath.name}")

if __name__ == "__main__":
    evaluate_and_plot()