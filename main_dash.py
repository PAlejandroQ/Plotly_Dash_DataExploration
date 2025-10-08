import dash
from dash import html, dcc, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from anomaly_detection import TimeSeriesAnomalyDetector
import uuid

# Application configuration
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "FILE LABEL"

def initialize_detector_with_csv():
    """
    Initializes the detector by loading data from CSV using Polars.
    """
    detector = TimeSeriesAnomalyDetector()

    # Load data from CSV with multiple series
    csv_path = 'csv_to_dash/multi_series_test.csv'
    try:
        series_options = detector.load_series_from_csv(csv_path, target_col='Value')
        print(f"Loaded {len(series_options)} series from CSV")
        for option in series_options:
            print(f"  - {option['label']} (WebId: {option['value']})")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Generating fallback data...")
        # Fallback to synthetic data if CSV loading fails
        series_options = generate_fallback_options(detector)

    # Apply Isolation Forest to all loaded series
    for series_name in detector.dataframes.keys():
        try:
            detector.apply_isolation_forest(
                series_name=series_name,
                target_col='Value',
                n_estimators=100,
                contamination=0.01  # 1% expected anomalies
            )
            print(f"Applied Isolation Forest to {series_name}")
        except Exception as e:
            print(f"Error processing {series_name}: {e}")

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
detector, series_options = initialize_detector_with_csv()

def create_graph_panel(panel_id):
    """
    Creates a template for a dynamic graph panel.

    Args:
        panel_id: Unique ID for the panel

    Returns:
        A dbc.Row component with the complete panel
    """
    return dbc.Row([
        # Panel header with delete button
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5(f"Visualization Panel #{panel_id}", className="d-inline-block me-2"),
                    dbc.Button(
                        "×",
                        id={'type': 'delete-graph-button', 'index': panel_id},
                        color="danger",
                        size="sm",
                        className="float-end",
                        style={"fontSize": "18px", "padding": "0 8px"}
                    )
                ]),
                dbc.CardBody([
                    # Panel controls
                    dbc.Row([
                        dbc.Col([
                            html.H6("Time Series"),
                            dcc.Checklist(
                                id={'type': 'series-selector-checklist', 'index': panel_id},
                                options=series_options,
                                value=[series_options[0]['value']] if series_options else [],  # Default first series
                                labelStyle={'display': 'block', 'margin-bottom': '5px', 'fontSize': '12px'},
                                inputStyle={"margin-right": "5px"}
                            )
                        ], width=6),
                        dbc.Col([
                            html.H6("Detection Methods"),
                            dcc.Checklist(
                                id={'type': 'methods-selector-checklist', 'index': panel_id},
                                options=[
                                    {'label': 'Isolation Forest', 'value': 'IF'}
                                ],
                                value=['IF'],
                                labelStyle={'display': 'block', 'margin-bottom': '5px', 'fontSize': '12px'},
                                inputStyle={"margin-right": "5px"}
                            )
                        ], width=6)
                    ]),

                    html.Hr(),

                    # Graph area
                    dcc.Loading(
                        id={'type': 'loading-plot', 'index': panel_id},
                        type="default",
                        children=[
                            dcc.Graph(
                                id={'type': 'anomalies-graph', 'index': panel_id},
                                style={'height': '400px'},
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ]
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ])

# Initial layout with one default panel
initial_panel_id = str(uuid.uuid4())[:8]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("FILE LABEL", className="text-center my-4", style={"color": "#2c3e50"}),
            html.Hr()
        ], width=12)
    ]),

    dcc.Store(id='graph-store', data=[{
        'id': initial_panel_id,
        'series': [series_options[0]['value']] if series_options else [],
        'methods': ['IF']
    }]),

    # Container for dynamic graphs (no height restrictions)
    html.Div(
        id='graph-container',
        children=[create_graph_panel(initial_panel_id)],
        style={"width": "100%"}
    ),

    # Button to duplicate panels (outside any Row/Col that might collapse)
    html.Div(
        dbc.Button(
            "➕ New Panel",
            id='add-graph-button',
            color="success",
            className="w-100",
            style={"marginTop": "180px"} # Adjusted to position the button lower
        ),
        style={"marginBottom": "40px"}
    )
], fluid=True, className="p-4")

@app.callback(
    [Output('graph-container', 'children'),
     Output('graph-store', 'data')],
    [Input('add-graph-button', 'n_clicks'),
     Input({'type': 'delete-graph-button', 'index': ALL}, 'n_clicks')],
    [State('graph-store', 'data')],
    prevent_initial_call=True
)
def manage_graph_panels(add_clicks, delete_clicks, current_panels):
    """
    Callback to manage the creation and deletion of panels.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id']

    # Determine which action was performed
    if 'add-graph-button' in trigger_id:
        # Add new panel
        new_panel_id = str(uuid.uuid4())[:8]
        current_panels.append({
            'id': new_panel_id,
            'series': [series_options[0]['value']] if series_options else [],
            'methods': ['IF']
        })
    else:
        # Delete specific panel
        # Find which delete button was clicked
        for i, click_count in enumerate(delete_clicks):
            if click_count and click_count > 0:
                # This is the panel to delete
                if i < len(current_panels):
                    current_panels.pop(i)
                break

    # Ensure at least one panel exists
    if not current_panels:
        new_panel_id = str(uuid.uuid4())[:8]
        current_panels = [{
            'id': new_panel_id,
            'series': [series_options[0]['value']] if series_options else [],
            'methods': ['IF']
        }]

    # Create components for all panels
    panel_components = [create_graph_panel(panel['id']) for panel in current_panels]

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
            showlegend=True,
            height=400
        )
        return fig

    # Create graph for this specific panel
    return detector.plot_multiple_series(selected_series, 'Value', selected_methods)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
