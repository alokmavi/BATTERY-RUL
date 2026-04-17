import torch
import torch.nn as nn

class BatteryHealthCNN(nn.Module):
    def __init__(self, sequence_length: int = 50, num_features: int = 3):
        super().__init__()
        
        # PyTorch Conv1d expects input shape: (Batch, Channels, Length)
        # Here, our 3 features (voltage, current, temp) act as the input channels
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate the flattened size dynamically based on sequence_length
        # Two MaxPool1d layers with kernel=2 will divide the length by 4 (50 // 4 = 12)
        flattened_length = sequence_length // 4
        self.flattened_size = 32 * flattened_length
        
        self.regressor = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Defensive: randomly zero elements to prevent overfitting
            nn.Linear(64, 1) # Single continuous output: Battery Capacity (Ah)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x arrives as (Batch, Sequence_Length, Features)
        # Permute it to (Batch, Features, Sequence_Length) to satisfy Conv1D
        x = x.permute(0, 2, 1)
        
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        
        return x.squeeze(-1)

def get_compute_device() -> torch.device:
    """Hardware-aware device selection targeting Apple Silicon."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

if __name__ == "__main__":
    device = get_compute_device()
    print(f"Targeting compute node: {device}")
    
    # Run a dummy tensor through the network to verify the architecture
    dummy_input = torch.randn(32, 50, 3).to(device) # Batch of 32
    model = BatteryHealthCNN().to(device)
    
    try:
        dummy_output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {dummy_output.shape}")
    except RuntimeError as shapeErr:
        print(f"Architecture shape mismatch: {shapeErr}")