#!/usr/bin/env python3
"""
Script to generate a test CSV file with multiple time series,
each with different WebId, Id, Name.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_multi_series_csv():
    """
    Generates a CSV with multiple time series to test the bulk loading functionality
    with Polars.
    """
    # Configuration
    start_date = datetime(2024, 1, 1)
    num_points_per_series = 500
    timestamps = [start_date + timedelta(minutes=i) for i in range(num_points_per_series)]

    # Define different series
    series_config = [
        {'web_id': 'WEB001', 'id': 'TEMP001', 'name': 'Temperature_Sensor_A'},
        {'web_id': 'WEB002', 'id': 'TEMP002', 'name': 'Temperature_Sensor_B'},
        {'web_id': 'WEB003', 'id': 'PRESS001', 'name': 'Pressure_Sensor_A'},
        {'web_id': 'WEB004', 'id': 'PRESS002', 'name': 'Pressure_Sensor_B'},
        {'web_id': 'WEB005', 'id': 'FLOW001', 'name': 'Flow_Rate_Sensor'},
    ]

    all_data = []

    np.random.seed(42)  # For reproducibility

    for config in series_config:
        web_id = config['web_id']
        id_ = config['id']
        name = config['name']

        # Generate different data based on sensor type
        if 'TEMP' in id_:
            # Temperature: sinusoidal pattern with noise
            base_value = 25 if 'A' in name else 30  # Different baselines
            values = (base_value +
                     5 * np.sin(2 * np.pi * np.arange(num_points_per_series) / 50) +
                     np.random.normal(0, 1, num_points_per_series))

            # Add anomalies
            anomaly_indices = np.random.choice(num_points_per_series, size=5, replace=False)
            values[anomaly_indices] += np.random.choice([-8, 8], size=5)

        elif 'PRESS' in id_:
            # Pressure: more stable pattern with level changes
            base_value = 100 if 'A' in name else 120
            values = base_value + np.random.normal(0, 2, num_points_per_series)

            # Level changes
            values[200:300] += 15
            values[400:] += 10

            # Anomalies
            anomaly_indices = np.random.choice(num_points_per_series, size=3, replace=False)
            values[anomaly_indices] += np.random.choice([-25, 25], size=3)

        elif 'FLOW' in id_:
            # Flow: pattern with trend and variability
            values = (50 +
                     0.02 * np.arange(num_points_per_series) +  # Trend
                     10 * np.sin(2 * np.pi * np.arange(num_points_per_series) / 100) +
                     np.random.normal(0, 3, num_points_per_series))

            # Anomalies
            anomaly_indices = np.random.choice(num_points_per_series, size=4, replace=False)
            values[anomaly_indices] += np.random.choice([-20, 20], size=4)

        # Create records for this series
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            all_data.append({
                'WebId': web_id,
                'Id': id_,
                'Name': name,
                'Timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z',
                'Value': round(value, 2)
            })

    # Create DataFrame and save as CSV
    df = pd.DataFrame(all_data)

    # Shuffle data to simulate an unordered file
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = 'csv_to_dash/multi_series_test.csv'
    df.to_csv(output_path, index=False)

    print(f"CSV file generated: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Generated series: {len(series_config)}")
    print("\nSummary by series:")
    for config in series_config:
        count = len(df[(df['WebId'] == config['web_id']) &
                       (df['Id'] == config['id']) &
                       (df['Name'] == config['name'])])
        print(f"  {config['web_id']} - {config['id']}_{config['name']}: {count} records")

    return output_path

if __name__ == '__main__':
    generate_multi_series_csv()
