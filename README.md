# Time Series Anomaly Detection

This project implements a flexible and extensible class for time series anomaly detection, initially using Isolation Forest, but designed to easily support other detection methods.

## Features

- **Anomaly Detection**: Implements Isolation Forest to identify outliers in time series
- **Interactive Visualization**: Uses Plotly to create interactive charts with highlighted anomalies
- **Efficient Bulk Loading**: Uses Polars to process large (GB) CSV files with multiple series
- **Dynamic Dashboard**: Visualization panels can be duplicated and deleted in real time
- **Extensible**: Modular architecture allows easy addition of new detection methods
- **Automatic Data Cleaning**: Automatically handles problematic numeric formats
- **Multiple Series**: Supports analysis of multiple time series simultaneously

## Project Structure

```
DataExplorationFieldLabel/
├── main_dash.py              # Dash web app with dynamic panels
├── anomaly_detection.py      # Main TimeSeriesAnomalyDetector class
├── data_explorer.ipynb       # Notebook with usage examples
├── csv/
│   └── time-series.csv       # Original example data
├── csv_to_dash/
│   └── multi_series_test.csv # Test data with multiple series (WebId, Id, Name)
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Installation

1. **Clone or download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or using conda:
   ```bash
   conda install pandas numpy scikit-learn plotly dash dash-bootstrap-components
   ```

## Basic Usage

```python
from anomaly_detection import TimeSeriesAnomalyDetector
import pandas as pd

# Load data
df = pd.read_csv('csv/time-series.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Initialize detector
detector = TimeSeriesAnomalyDetector()

# Add time series
detector.add_series('MySeries', df)

# Apply anomaly detection
detector.apply_isolation_forest('MySeries', 'Value')

# Visualize results
fig = detector.plot_anomalies('MySeries', 'Value', ['IF'])
fig.show()
```

## Bulk Data Loading with Polars

For large CSV files (GB), the class includes the `load_series_from_csv()` method using Polars for efficient loading:

```python
from anomaly_detection import TimeSeriesAnomalyDetector

detector = TimeSeriesAnomalyDetector()

# Load multiple series grouped by WebId, Id, Name
options = detector.load_series_from_csv('csv_to_dash/multi_series_test.csv')

print(f"Loaded {len(options)} series:")
for option in options:
    print(f"  {option['label']} -> WebId: {option['value']}")
```

**Bulk loading features:**
- **Polars**: Ultra-fast processing of large files
- **Automatic grouping**: Series grouped by WebId, Id, Name
- **Automatic conversion**: Timestamps and data types
- **Dash options**: Returns ready-to-use format for selectors

## Dash Web App with Dynamic Panels

To run the full interactive web app:

```bash
python main_dash.py
```

The app will be available at `http://127.0.0.1:8050/`

### Advanced Dashboard Features

**Dynamic Panels:**
- ➕ **"New Panel" button**: Create new independent visualization panels
- ❌ **"×" button on each panel**: Delete specific panels
- **dcc.Store**: Maintains the state of all active panels
- **MATCH/ALL Callbacks**: Independent update of each panel

**Interactive Visualization:**
- Sidebar to select time series and detection methods
- Interactive Plotly visualization in each panel
- Support for multiple series per panel
- Loading indicator during processing
- Elegant Bootstrap interface

**Demo Data:**
- **TEMP001_Temperature_Sensor_A**: Sinusoidal pattern with anomalies
- **TEMP002_Temperature_Sensor_B**: Linear trend with seasonality
- **PRESS001_Pressure_Sensor_A**: Level shifts
- **PRESS002_Pressure_Sensor_B**: Moderate variability
- **FLOW001_Flow_Rate_Sensor**: Trend with high variability

## TimeSeriesAnomalyDetector Class

### Main Methods

- `__init__(series_data=None)`: Initializes the detector
- `add_series(name, df)`: Adds a new time series
- `load_series_from_csv(file_path, target_col='Value')`: Bulk load series from CSV using Polars
- `apply_isolation_forest(series_name, target_col, n_estimators=100, contamination=0.01)`: Apply Isolation Forest
- `apply_method(series_name, method_name, **kwargs)`: Placeholder for future methods
- `plot_anomalies(series_name, target_col, methods_to_plot)`: Create interactive visualization
- `plot_multiple_series(series_names, target_col, methods_to_plot)`: Combined visualization of multiple series

### Technical Features

- **Feature Engineering**: Automatically creates lags (t-1, t-2, t-3) to improve detection
- **Data Cleaning**: Automatically handles values with thousands separators
- **Validation**: Checks for existence of series and columns before processing
- **Flexibility**: Supports different parameters for detection algorithms

## Notebook Examples

The `data_explorer.ipynb` file contains complete examples demonstrating:

1. Data loading and exploration
2. Detector initialization
3. Application of detection methods
4. Visualization of results
5. Working with multiple series

## System Extension

To add new detection methods, implement new methods following the `apply_isolation_forest` pattern:

```python
def apply_one_class_svm(self, series_name: str, target_col: str, **kwargs) -> None:
    # Method implementation
    # ...
    # Save results in self.results[series_name]
```

## Example Data

The data in `csv/time-series.csv` contains a real time series with timestamps and numeric values. Some values may have special formats that are automatically cleaned by the class.

## Requirements

- Python 3.7+
- pandas
- numpy
- polars
- scikit-learn
- plotly
- dash
- dash-bootstrap-components

<!-- ## License

This project is open source and may be freely used for educational and commercial purposes. -->
