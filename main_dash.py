import dash
from dash import html, dcc, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from anomaly_detection import TimeSeriesAnomalyDetector
import plotly.graph_objects as go
import ast
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application configuration
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "FIELD LABEL"

def initialize_detector():
    """
    Initializes the detector by loading data from either CSV or Parquet files based on environment configuration.
    """
    detector = TimeSeriesAnomalyDetector()

    # Get configuration from environment variables
    data_source = os.getenv('DATA_SOURCE', 'PARQUET').upper()
    parquet_folder = os.getenv('PARQUET_FOLDER', 'parquets')
    csv_file = os.getenv('CSV_FILE', 'csv_to_dash/multi_series_test.csv')

    series_options = []

    if data_source == 'PARQUET':
        print("Loading data from Parquet files...")
        try:
            series_options = detector.load_series_from_parquet(parquet_folder, target_col='Value')
            print(f"Loaded {len(series_options)} series from Parquet files")
            for option in series_options:
                print(f"  - {option['label']}")
        except Exception as e:
            print(f"Error loading Parquet files: {e}")
            print("Falling back to CSV...")
            data_source = 'CSV'

    if data_source == 'CSV':
        print("Loading data from CSV file...")
        try:
            series_options = detector.load_series_from_csv(csv_file, target_col='Value')
            print(f"Loaded {len(series_options)} series from CSV")
            for option in series_options:
                print(f"  - {option['label']} (WebId: {option['value']})")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Generating fallback data...")
            # Fallback to synthetic data if both sources fail
            series_options = generate_fallback_options(detector)

    # Apply Isolation Forest to all loaded series
    # for series_name in detector.dataframes.keys():
    #     try:
    #         detector.apply_isolation_forest(
    #             series_name=series_name,
    #             target_col='Value',
    #             n_estimators=100,
    #             contamination=0.01  # 1% expected anomalies
    #         )
    #         print(f"Applied Isolation Forest to {series_name}")
    #     except Exception as e:
    #         print(f"Error processing {series_name}: {e}")

    return detector, series_options

def generate_fallback_options(detector):
    """
    Generates fallback options if CSV loading fails.
    """
    # Simple synthetic data as fallback
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)]

    # Simple series
    values = 10 + np.random.normal(0, 1, 100)
    df = pd.DataFrame({'Value': values}, index=timestamps)

    detector.add_series('FALLBACK001', df)

    return [{'label': 'Fallback_Series', 'value': 'FALLBACK001'}]

# Initialize detector and global options
detector, series_options = initialize_detector()

def create_graph_panel(panel_id, selected_series=None, selected_methods=None):
    """
    Creates a template for a dynamic graph panel.

    Args:
        panel_id: Unique ID for the panel
        selected_series: List of selected series (optional)
        selected_methods: List of selected methods (optional)

    Returns:
        A dbc.Row component with the complete panel
    """
    # Default values if not provided
    if selected_series is None:
        selected_series = [series_options[0]['value']] if series_options else []
    if selected_methods is None:
        selected_methods = ['IF']

    # Pre-compute initial figure so each panel renders without interaction
    if selected_series:
        initial_figure = detector.plot_multiple_series(selected_series, 'Value', selected_methods)
    else:
        initial_figure = go.Figure()
        initial_figure.update_layout(
            title="Select at least one time series",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )

    return dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H6("Time Series", className="fw-bold"),
                                    dcc.Checklist(
                                        id={'type': 'series-selector-checklist', 'index': panel_id},
                                        options=series_options,
                                        value=selected_series,
                                        labelStyle={'display': 'block', 'margin-bottom': '5px', 'fontSize': '12px'},
                                        inputStyle={"margin-right": "5px"},
                                        style={"maxHeight": "220px", "overflowY": "auto"}
                                    ),
                                    html.Hr(className="my-3"),
                                    html.H6("Detection Methods", className="fw-bold"),
                                    dcc.Checklist(
                                        id={'type': 'methods-selector-checklist', 'index': panel_id},
                                        options=[{'label': 'Isolation Forest', 'value': 'IF'}],
                                        value=selected_methods,
                                        labelStyle={'display': 'block', 'margin-bottom': '5px', 'fontSize': '12px'},
                                        inputStyle={"margin-right": "5px"}
                                    ),
                                    html.Hr(className="my-3"),
                                    dbc.Button(
                                        "×",
                                        id={'type': 'delete-graph-button', 'index': panel_id},
                                        color="danger",
                                        size="sm",
                                        style={"fontSize": "18px", "padding": "0 8px"}
                                    )
                                ]),
                                className="h-100"
                            )
                        ], width=3),
                        dbc.Col([
                            dcc.Loading(
                                id={'type': 'loading-plot', 'index': panel_id},
                                type="default",
                                children=[
                                    dcc.Graph(
                                        id={'type': 'anomalies-graph', 'index': panel_id},
                                        style={'height': '100%', 'width': '100%'},
                                        config={'displayModeBar': True, 'displaylogo': False},
                                        figure=initial_figure
                                    )
                                ]
                            )
                        ], width=8)
                    ], className="g-3", align="stretch")
                ]),
                className="mb-4",
                style={"height": "100%", "overflow": "hidden"}
            )
        ], width=12)
    ])

# Initial layout with one default panel
initial_panel_id = str(uuid.uuid4())[:8]
default_series = [series_options[0]['value']] if series_options else []

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("FIELD LABEL", className="text-center my-4", style={"color": "#2c3e50"}),
            html.Hr()
        ], width=12)
    ]),

    # Store to manage panel states (series and methods selections)
    dcc.Store(id='graph-store', data=[{
        'id': initial_panel_id,
        'series': default_series,
        'methods': ['IF']
    }]),

    # Dynamic graph container
    html.Div(
        id='graph-container',
        children=[create_graph_panel(initial_panel_id, default_series, ['IF'])],
        style={"width": "100%"}
    ),

    # Button to duplicate panels
    html.Div(
        dbc.Button(
            "➕ New Panel",
            id='add-graph-button',
            color="success",
            className="w-100",
            style={"marginTop": "180px"}
        ),
        style={"marginBottom": "40px"}
    )
], fluid=True, className="p-4")

# Callback to update panel selections in store
@app.callback(
    [Output('graph-container', 'children'),
     Output('graph-store', 'data', allow_duplicate=True)],
    [Input('add-graph-button', 'n_clicks'),
     Input({'type': 'delete-graph-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'series-selector-checklist', 'index': ALL}, 'value'),
     Input({'type': 'methods-selector-checklist', 'index': ALL}, 'value')],
    [State('graph-store', 'data')],
    prevent_initial_call=True
)
def manage_graph_panels(add_clicks, delete_clicks, series_values, methods_values, current_panels):
    """
    Callback to manage the creation and deletion of panels.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered = ctx.triggered[0]['prop_id']
    raw_trigger = triggered.split('.')[0] if triggered else ''
    if raw_trigger and raw_trigger.startswith('{'):
        trigger_id = ast.literal_eval(raw_trigger)
    else:
        trigger_id = raw_trigger

    # Normalize current state arrays
    if current_panels is None:
        current_panels = []
    series_values = series_values or []
    methods_values = methods_values or []

    # Update stored selections for current panels
    for idx, panel in enumerate(current_panels):
        if idx < len(series_values):
            panel['series'] = series_values[idx] or []
        if idx < len(methods_values):
            panel['methods'] = methods_values[idx] or []

    # Determine which action was performed
    if trigger_id == 'add-graph-button':
        new_panel_id = str(uuid.uuid4())[:8]
        current_panels.append({
            'id': new_panel_id,
            'series': default_series.copy(),
            'methods': ['IF']
        })
    elif isinstance(trigger_id, dict) and trigger_id.get('type') == 'delete-graph-button':
        delete_index = trigger_id['index']
        current_panels = [panel for panel in current_panels if panel['id'] != delete_index]

    # Ensure at least one panel exists
    if not current_panels:
        new_panel_id = str(uuid.uuid4())[:8]
        current_panels = [{
            'id': new_panel_id,
            'series': default_series.copy(),
            'methods': ['IF']
        }]

    # Create components for all panels using their current state
    panel_components = []
    for panel in current_panels:
        panel_id = panel['id']
        selected_series = panel.get('series', []) or default_series.copy()
        selected_methods = panel.get('methods', ['IF']) or ['IF']
        panel_components.append(create_graph_panel(panel_id, selected_series, selected_methods))

    return panel_components, current_panels

@app.callback(
    Output({'type': 'anomalies-graph', 'index': MATCH}, 'figure'),
    [Input({'type': 'series-selector-checklist', 'index': MATCH}, 'value'),
     Input({'type': 'methods-selector-checklist', 'index': MATCH}, 'value')],
    [State({'type': 'anomalies-graph', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_individual_graph(selected_series, selected_methods, graph_id):
    """
    Callback to update individual graphs using MATCH pattern.
    """
    panel_id = graph_id['index']

    if not selected_series:
        # If no series are selected, show an empty graph
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="Select at least one time series",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )
        return fig

    # Create graph for this specific panel
    return detector.plot_multiple_series(selected_series, 'Value', selected_methods)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
