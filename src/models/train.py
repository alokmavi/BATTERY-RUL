import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Resolve root to import our ingestion and model modules
projectRoot = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(projectRoot))

from src.ingestion.sequence_builder import build_feature_tensors
from src.models.cnn_estimator import BatteryHealthCNN, get_compute_device

def prepare_dataloaders(csvPath: Path, batchSize: int = 32):
    """Loads feature tensors and constructs chronological data loaders."""
    # We call the sequence builder directly so we don't have to manage raw .npy files on disk
    X_np, y_np = build_feature_tensors(csvPath, sequenceLength=50, stepSize=10)
    
    # Strict Time-Series Split (Train on past, Test on future)
    splitIndex = int(len(X_np) * 0.8)
    
    # Convert numpy arrays to float32 tensors (required for PyTorch weights)
    X_train = torch.tensor(X_np[:splitIndex], dtype=torch.float32)
    y_train = torch.tensor(y_np[:splitIndex], dtype=torch.float32)
    X_test = torch.tensor(X_np[splitIndex:], dtype=torch.float32)
    y_test = torch.tensor(y_np[splitIndex:], dtype=torch.float32)
    
    trainDataset = TensorDataset(X_train, y_train)
    testDataset = TensorDataset(X_test, y_test)
    
    # Shuffling training batches is fine since the sequences themselves are chronological internally
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
    
    return trainLoader, testLoader

def execute_training_run(epochs: int = 20):
    device = get_compute_device()
    processedCsvPath = projectRoot / "data" / "processed" / "B0005_discharge.csv"
    
    trainLoader, testLoader = prepare_dataloaders(processedCsvPath)
    
    model = BatteryHealthCNN().to(device)
    
    # Regression problem: Mean Squared Error (MSE) is the standard loss metric
    lossFunction = nn.MSELoss()
    # Adam optimizer with a conservative learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Beginning training on {device}...\n")
    
    for epoch in range(1, epochs + 1):
        model.train()
        totalTrainLoss = 0.0
        
        for batchX, batchY in trainLoader:
            batchX, batchY = batchX.to(device), batchY.to(device)
            
            optimizer.zero_grad()
            predictions = model(batchX)
            
            loss = lossFunction(predictions, batchY)
            loss.backward()
            optimizer.step()
            
            totalTrainLoss += loss.item()
            
        avgTrainLoss = totalTrainLoss / len(trainLoader)
        
        # Validation Phase
        model.eval()
        totalTestLoss = 0.0
        with torch.no_grad():
            for batchX, batchY in testLoader:
                batchX, batchY = batchX.to(device), batchY.to(device)
                predictions = model(batchX)
                loss = lossFunction(predictions, batchY)
                totalTestLoss += loss.item()
                
        avgTestLoss = totalTestLoss / len(testLoader)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss (MSE): {avgTrainLoss:.4f} | Test Loss: {avgTestLoss:.4f}")

    # Save the trained weights
    modelSavePath = projectRoot / "src" / "models" / "battery_cnn_weights.pth"
    torch.save(model.state_dict(), modelSavePath)
    print(f"\nTraining complete. Model weights saved to {modelSavePath.name}")

if __name__ == "__main__":
    try:
        execute_training_run(epochs=20)
    except Exception as err:
        print(f"Training pipeline failed: {err}")