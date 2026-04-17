import scipy.io
import pandas as pd
from pathlib import Path

def parse_battery_telemetry(matFilePath: Path, targetCsvPath: Path) -> None:
    if not matFilePath.exists():
        raise FileNotFoundError(f"Source telemetry missing at: {matFilePath}")

    try:
        # SciPy loadmat returns a dictionary. The key matches the filename (e.g., 'B0005').
        batteryIdentifier = matFilePath.stem
        rawMatData = scipy.io.loadmat(str(matFilePath))
        cycleMatrix = rawMatData[batteryIdentifier][0, 0]['cycle'][0]
        
        extractedTelemetry = []
        
        for cycle in cycleMatrix:
            operationType = cycle['type'][0]
            if operationType != 'discharge':
                continue
            
            sensorData = cycle['data'][0, 0]
            timeSequence = sensorData['Time'][0]
            voltageSequence = sensorData['Voltage_measured'][0]
            currentSequence = sensorData['Current_measured'][0]
            tempSequence = sensorData['Temperature_measured'][0]
            
            capacityAvailable = sensorData['Capacity'][0][0] if 'Capacity' in sensorData.dtype.names else None
            
            for index, timestamp in enumerate(timeSequence):
                extractedTelemetry.append({
                    'timestamp_s': timestamp,
                    'voltage_v': voltageSequence[index],
                    'current_a': currentSequence[index],
                    'temperature_c': tempSequence[index],
                    'capacity_ah': capacityAvailable
                })
        
        telemetryDataFrame = pd.DataFrame(extractedTelemetry)
        targetCsvPath.parent.mkdir(parents=True, exist_ok=True)
        telemetryDataFrame.to_csv(targetCsvPath, index=False)
        
    except KeyError as missingKeyErr:
        raise ValueError(f"Unexpected .mat structure. Missing key: {missingKeyErr}")
    except Exception as parseErr:
        raise RuntimeError(f"Failed to parse matrix data: {parseErr}")

if __name__ == "__main__":
    projectRoot = Path(__file__).resolve().parent.parent.parent
    rawTelemetryPath = projectRoot / "data" / "raw" / "B0005.mat"
    processedOutputPath = projectRoot / "data" / "processed" / "B0005_discharge.csv"
    
    parse_battery_telemetry(rawTelemetryPath, processedOutputPath)