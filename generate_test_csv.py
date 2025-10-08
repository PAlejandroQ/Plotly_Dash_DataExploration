#!/usr/bin/env python3
"""
Script para generar un archivo CSV de prueba con múltiples series de tiempo,
cada una con diferentes WebId, Id, Name.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_multi_series_csv():
    """
    Genera un CSV con múltiples series de tiempo para probar la funcionalidad
    de carga masiva con Polars.
    """
    # Configuración
    start_date = datetime(2024, 1, 1)
    num_points_per_series = 500
    timestamps = [start_date + timedelta(minutes=i) for i in range(num_points_per_series)]

    # Definir las diferentes series
    series_config = [
        {'web_id': 'WEB001', 'id': 'TEMP001', 'name': 'Temperature_Sensor_A'},
        {'web_id': 'WEB002', 'id': 'TEMP002', 'name': 'Temperature_Sensor_B'},
        {'web_id': 'WEB003', 'id': 'PRESS001', 'name': 'Pressure_Sensor_A'},
        {'web_id': 'WEB004', 'id': 'PRESS002', 'name': 'Pressure_Sensor_B'},
        {'web_id': 'WEB005', 'id': 'FLOW001', 'name': 'Flow_Rate_Sensor'},
    ]

    all_data = []

    np.random.seed(42)  # Para reproducibilidad

    for config in series_config:
        web_id = config['web_id']
        id_ = config['id']
        name = config['name']

        # Generar datos diferentes según el tipo de sensor
        if 'TEMP' in id_:
            # Temperatura: patrón sinusoidal con ruido
            base_value = 25 if 'A' in name else 30  # Diferentes baselines
            values = (base_value +
                     5 * np.sin(2 * np.pi * np.arange(num_points_per_series) / 50) +
                     np.random.normal(0, 1, num_points_per_series))

            # Agregar anomalías
            anomaly_indices = np.random.choice(num_points_per_series, size=5, replace=False)
            values[anomaly_indices] += np.random.choice([-8, 8], size=5)

        elif 'PRESS' in id_:
            # Presión: patrón más estable con cambios de nivel
            base_value = 100 if 'A' in name else 120
            values = base_value + np.random.normal(0, 2, num_points_per_series)

            # Cambios de nivel
            values[200:300] += 15
            values[400:] += 10

            # Anomalías
            anomaly_indices = np.random.choice(num_points_per_series, size=3, replace=False)
            values[anomaly_indices] += np.random.choice([-25, 25], size=3)

        elif 'FLOW' in id_:
            # Flujo: patrón con tendencia y variabilidad
            values = (50 +
                     0.02 * np.arange(num_points_per_series) +  # Tendencia
                     10 * np.sin(2 * np.pi * np.arange(num_points_per_series) / 100) +
                     np.random.normal(0, 3, num_points_per_series))

            # Anomalías
            anomaly_indices = np.random.choice(num_points_per_series, size=4, replace=False)
            values[anomaly_indices] += np.random.choice([-20, 20], size=4)

        # Crear registros para esta serie
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            all_data.append({
                'WebId': web_id,
                'Id': id_,
                'Name': name,
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Value': round(value, 2)
            })

    # Crear DataFrame y guardar como CSV
    df = pd.DataFrame(all_data)

    # Mezclar los datos para simular un archivo desordenado
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = 'csv_to_dash/multi_series_test.csv'
    df.to_csv(output_path, index=False)

    print(f"Archivo CSV generado: {output_path}")
    print(f"Total de registros: {len(df)}")
    print(f"Series generadas: {len(series_config)}")
    print("\nResumen por serie:")
    for config in series_config:
        count = len(df[(df['WebId'] == config['web_id']) &
                       (df['Id'] == config['id']) &
                       (df['Name'] == config['name'])])
        print(f"  {config['web_id']} - {config['id']}_{config['name']}: {count} registros")

    return output_path

if __name__ == '__main__':
    generate_multi_series_csv()
