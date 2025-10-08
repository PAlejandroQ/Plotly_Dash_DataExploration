import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from anomaly_detection import TimeSeriesAnomalyDetector

# Configuración de la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "FILE LABEL"

def generate_synthetic_data():
    """
    Genera datos sintéticos de series de tiempo con anomalías claras.
    """
    # Crear timestamps para 1000 puntos de datos
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i) for i in range(1000)]

    np.random.seed(42)  # Para reproducibilidad

    # Serie A: Patrón sinusoidal con ruido y algunas anomalías
    time_values = np.arange(1000)
    serie_a_values = 10 + 5 * np.sin(2 * np.pi * time_values / 100) + np.random.normal(0, 0.5, 1000)

    # Agregar anomalías claras en Serie A
    anomaly_indices_a = [200, 400, 600, 800]
    for idx in anomaly_indices_a:
        serie_a_values[idx] += np.random.choice([-15, 15])  # Anomalías grandes

    # Serie B: Tendencia lineal con estacionalidad y anomalías
    serie_b_values = 20 + 0.01 * time_values + 3 * np.sin(2 * np.pi * time_values / 50) + np.random.normal(0, 0.8, 1000)

    # Agregar anomalías en Serie B
    anomaly_indices_b = [150, 350, 550, 750, 900]
    for idx in anomaly_indices_b:
        serie_b_values[idx] += np.random.choice([-12, 12])  # Anomalías grandes

    # Serie C: Patrón más complejo con cambios de nivel
    serie_c_values = 15 + np.random.normal(0, 1, 1000)
    # Cambios de nivel
    serie_c_values[300:500] += 8
    serie_c_values[700:] += 5

    # Agregar anomalías en Serie C
    anomaly_indices_c = [100, 450, 800]
    for idx in anomaly_indices_c:
        serie_c_values[idx] += np.random.choice([-20, 20])

    # Crear DataFrames
    df_a = pd.DataFrame({'Value': serie_a_values}, index=timestamps)
    df_b = pd.DataFrame({'Value': serie_b_values}, index=timestamps)
    df_c = pd.DataFrame({'Value': serie_c_values}, index=timestamps)

    return {
        'SerieA': df_a,
        'SerieB': df_b,
        'SerieC': df_c
    }

def initialize_detector():
    """
    Inicializa el detector con datos sintéticos y aplica Isolation Forest.
    """
    # Generar datos
    synthetic_data = generate_synthetic_data()

    # Inicializar detector
    detector = TimeSeriesAnomalyDetector(synthetic_data)

    # Aplicar Isolation Forest a todas las series
    for series_name in detector.dataframes.keys():
        detector.apply_isolation_forest(
            series_name=series_name,
            target_col='Value',
            n_estimators=100,
            contamination=0.01  # 1% de anomalías esperadas
        )

    return detector

# Inicializar el detector global
detector = initialize_detector()

# Layout de la aplicación
app.layout = dbc.Container([
    dbc.Row([
        # Título principal
        dbc.Col([
            html.H1("FILE LABEL", className="text-center my-4", style={"color": "#2c3e50"}),
            html.Hr()
        ], width=12)
    ]),

    dbc.Row([
        # Panel lateral de control (3 columnas)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Panel de Control", className="text-center")),
                dbc.CardBody([
                    # Selector de Series
                    html.H5("Seleccionar Series de Tiempo"),
                    dcc.Checklist(
                        id='series-selector-checklist',
                        options=[
                            {'label': 'Serie A (Sinusoidal)', 'value': 'SerieA'},
                            {'label': 'Serie B (Tendencia)', 'value': 'SerieB'},
                            {'label': 'Serie C (Cambios de Nivel)', 'value': 'SerieC'}
                        ],
                        value=['SerieA'],  # Serie A seleccionada por defecto
                        labelStyle={'display': 'block', 'margin-bottom': '10px'},
                        inputStyle={"margin-right": "10px"}
                    ),

                    html.Hr(),

                    # Selector de Métodos
                    html.H5("Seleccionar Métodos de Detección"),
                    dcc.Checklist(
                        id='methods-selector-checklist',
                        options=[
                            {'label': 'Isolation Forest (IF)', 'value': 'IF'}
                        ],
                        value=['IF'],  # IF seleccionado por defecto
                        labelStyle={'display': 'block', 'margin-bottom': '10px'},
                        inputStyle={"margin-right": "10px"}
                    ),

                    html.Hr(),

                    # Información adicional
                    html.Div([
                        html.H6("Información:"),
                        html.P("Selecciona una o más series de tiempo y métodos de detección. "
                              "Las anomalías detectadas se mostrarán como puntos rojos en el gráfico.",
                              className="text-muted small")
                    ])
                ])
            ], className="mb-4")
        ], width=3),

        # Área de gráfico (9 columnas)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Visualización de Anomalías", className="text-center")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-plot",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='anomalies-graph',
                                style={'height': '600px'},
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ]
                    )
                ])
            ])
        ], width=9)
    ])
], fluid=True, className="p-4")

@app.callback(
    Output('anomalies-graph', 'figure'),
    Input('series-selector-checklist', 'value'),
    Input('methods-selector-checklist', 'value')
)
def update_graph(selected_series, selected_methods):
    """
    Callback principal que actualiza el gráfico basado en las selecciones del usuario.
    """
    if not selected_series:
        # Si no hay series seleccionadas, mostrar gráfico vacío
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="Selecciona al menos una serie de tiempo",
            xaxis_title="Tiempo",
            yaxis_title="Valor",
            showlegend=True
        )
        return fig

    # Crear gráfico combinado para múltiples series
    return detector.plot_multiple_series(selected_series, 'Value', selected_methods)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
