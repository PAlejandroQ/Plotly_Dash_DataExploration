# Time Series Anomaly Detection

Este proyecto implementa una clase flexible y extensible para la detección de anomalías en series de tiempo, inicialmente utilizando Isolation Forest, pero diseñada para admitir fácilmente otros métodos de detección.

## Características

- **Detección de Anomalías**: Implementa Isolation Forest para identificar valores atípicos en series de tiempo
- **Visualización Interactiva**: Utiliza Plotly para crear gráficos interactivos con anomalías destacadas
- **Carga Masiva Eficiente**: Usa Polars para procesar archivos CSV grandes (GB) con múltiples series
- **Dashboard Dinámico**: Paneles de visualización duplicables y eliminables en tiempo real
- **Extensible**: Arquitectura modular que permite agregar nuevos métodos de detección fácilmente
- **Limpieza Automática de Datos**: Maneja automáticamente formatos numéricos problemáticos
- **Múltiples Series**: Soporta análisis de múltiples series de tiempo simultáneamente

## Estructura del Proyecto

```
DataExplorationFieldLabel/
├── main_dash.py              # Aplicación web Dash interactiva con paneles dinámicos
├── anomaly_detection.py      # Clase principal TimeSeriesAnomalyDetector
├── data_explorer.ipynb       # Notebook con ejemplos de uso
├── csv/
│   └── time-series.csv       # Datos de ejemplo originales
├── csv_to_dash/
│   └── multi_series_test.csv # Datos de prueba con múltiples series (WebId, Id, Name)
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Este archivo
```

## Instalación

1. **Clona o descarga el repositorio**
2. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
   O usando conda:
   ```bash
   conda install pandas numpy scikit-learn plotly dash dash-bootstrap-components
   ```

## Uso Básico

```python
from anomaly_detection import TimeSeriesAnomalyDetector
import pandas as pd

# Cargar datos
df = pd.read_csv('csv/time-series.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Inicializar detector
detector = TimeSeriesAnomalyDetector()

# Agregar serie de tiempo
detector.add_series('MiSerie', df)

# Aplicar detección de anomalías
detector.apply_isolation_forest('MiSerie', 'Value')

# Visualizar resultados
fig = detector.plot_anomalies('MiSerie', 'Value', ['IF'])
fig.show()
```

## Carga Masiva de Datos con Polars

Para archivos CSV grandes (GB), la clase incluye el método `load_series_from_csv()` que utiliza Polars para una carga eficiente:

```python
from anomaly_detection import TimeSeriesAnomalyDetector

detector = TimeSeriesAnomalyDetector()

# Cargar múltiples series agrupadas por WebId, Id, Name
options = detector.load_series_from_csv('csv_to_dash/multi_series_test.csv')

print(f"Cargadas {len(options)} series:")
for option in options:
    print(f"  {option['label']} -> WebId: {option['value']}")
```

**Características de carga masiva:**
- **Polars**: Procesamiento ultrarrápido de archivos grandes
- **Agrupación automática**: Series agrupadas por WebId, Id, Name
- **Conversión automática**: Timestamps y tipos de datos
- **Opciones para Dash**: Retorna formato listo para selectores

## Aplicación Web Dash con Paneles Dinámicos

Para ejecutar la aplicación web interactiva completa:

```bash
python main_dash.py
```

La aplicación estará disponible en `http://127.0.0.1:8050/`

### Características Avanzadas del Dashboard

**Paneles Dinámicos:**
- ➕ **Botón "Duplicar Panel"**: Crea nuevos paneles de visualización independientes
- ❌ **Botón "×" en cada panel**: Elimina paneles específicos
- **dcc.Store**: Mantiene el estado de todos los paneles activos
- **Callbacks MATCH/ALL**: Actualización independiente de cada panel

**Visualización Interactiva:**
- Panel lateral para seleccionar series de tiempo y métodos de detección
- Visualización interactiva con Plotly en cada panel
- Soporte para múltiples series simultáneamente por panel
- Indicador de carga durante el procesamiento
- Interfaz elegante con Bootstrap

**Datos de Demostración:**
- **TEMP001_Temperature_Sensor_A**: Patrón sinusoidal con anomalías
- **TEMP002_Temperature_Sensor_B**: Tendencia lineal con estacionalidad
- **PRESS001_Pressure_Sensor_A**: Patrón con cambios de nivel
- **PRESS002_Pressure_Sensor_B**: Variabilidad moderada
- **FLOW001_Flow_Rate_Sensor**: Tendencia con variabilidad alta

## Clase TimeSeriesAnomalyDetector

### Métodos Principales

- `__init__(series_data=None)`: Inicializa el detector
- `add_series(name, df)`: Agrega una nueva serie de tiempo
- `load_series_from_csv(file_path, target_col='Value')`: Carga masiva de series desde CSV usando Polars
- `apply_isolation_forest(series_name, target_col, n_estimators=100, contamination=0.01)`: Aplica Isolation Forest
- `apply_method(series_name, method_name, **kwargs)`: Placeholder para métodos futuros
- `plot_anomalies(series_name, target_col, methods_to_plot)`: Crea visualización interactiva
- `plot_multiple_series(series_names, target_col, methods_to_plot)`: Visualización combinada de múltiples series

### Características Técnicas

- **Feature Engineering**: Crea automáticamente lags (t-1, t-2, t-3) para mejorar la detección
- **Limpieza de Datos**: Maneja automáticamente valores con separadores de miles
- **Validación**: Verifica la existencia de series y columnas antes del procesamiento
- **Flexibilidad**: Soporta diferentes parámetros para algoritmos de detección

## Ejemplos en el Notebook

El archivo `data_explorer.ipynb` contiene ejemplos completos que demuestran:

1. Carga y exploración de datos
2. Inicialización del detector
3. Aplicación de métodos de detección
4. Visualización de resultados
5. Trabajo con múltiples series

## Extensión del Sistema

Para agregar nuevos métodos de detección, implementa nuevos métodos siguiendo el patrón de `apply_isolation_forest`:

```python
def apply_one_class_svm(self, series_name: str, target_col: str, **kwargs) -> None:
    # Implementación del método
    # ...
    # Guardar resultados en self.results[series_name]
```

## Datos de Ejemplo

Los datos en `csv/time-series.csv` contienen una serie de tiempo real con timestamps y valores numéricos. Algunos valores pueden tener formatos especiales que son limpiados automáticamente por la clase.

## Requisitos

- Python 3.7+
- pandas
- numpy
- polars
- scikit-learn
- plotly
- dash
- dash-bootstrap-components

<!-- ## Licencia

Este proyecto es de código abierto y puede ser utilizado libremente para fines educativos y comerciales. -->
