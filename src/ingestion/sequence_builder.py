import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def build_feature_tensors(
    csvPath: Path, 
    sequenceLength: int = 50, 
    stepSize: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D tabular telemetry into 3D tensors for sequential modeling.
    Enforces time-continuity within windows to prevent cross-cycle contamination.
    """
    if not csvPath.exists():
        raise FileNotFoundError(f"Processed telemetry not found at: {csvPath}")

    df = pd.read_csv(csvPath)
    
    # Isolate features and target
    featureColumns = ['voltage_v', 'current_a', 'temperature_c']
    targetColumn = 'capacity_ah'
    
    rawFeatures = df[featureColumns].to_numpy()
    rawTargets = df[targetColumn].to_numpy()
    timeSequence = df['timestamp_s'].to_numpy()

    # Z-Score Normalization (Standardization)
    featureMeans = np.mean(rawFeatures, axis=0)
    featureStds = np.std(rawFeatures, axis=0)
    
    # Defensive check to prevent division by zero on flatline sensors
    featureStds[featureStds == 0] = 1e-6 
    normalizedFeatures = (rawFeatures - featureMeans) / featureStds

    X_windows = []
    y_targets = []

    # Sliding window extraction
    for i in range(0, len(df) - sequenceLength, stepSize):
        windowEnd = i + sequenceLength
        
        # Boundary Check: If time drops within the window, a new cycle started. 
        # Discard this contaminated window.
        timeWindow = timeSequence[i:windowEnd]
        if np.any(np.diff(timeWindow) < 0):
            continue
            
        X_windows.append(normalizedFeatures[i:windowEnd])
        
        # The target is the capacity at the end of the observed sequence
        y_targets.append(rawTargets[windowEnd - 1])

    X_tensor = np.array(X_windows)
    y_tensor = np.array(y_targets)

    print(f"Extraction complete. Tensor shapes:")
    print(f"X (Features): {X_tensor.shape} -> (samples, sequence_length, metrics)")
    print(f"y (Target):   {y_tensor.shape} -> (samples, target_capacity)")

    return X_tensor, y_tensor

if __name__ == "__main__":
    projectRoot = Path(__file__).resolve().parent.parent.parent
    processedCsvPath = projectRoot / "data" / "processed" / "B0005_discharge.csv"
    
    try:
        X, y = build_feature_tensors(processedCsvPath, sequenceLength=50, stepSize=10)
        # We will save these arrays to disk in the next phase before training.
    except Exception as err:
        print(f"Sequence generation failed: {err}")