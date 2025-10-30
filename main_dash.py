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
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application configuration
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "FIELD LABEL"

# Load metadata descriptions
def load_metadata_descriptions():
    """Load descriptions from metadata.json file."""
    descriptions = {}
    try:
        # Try to load from the parquet folder first
        parquet_folder = os.getenv('PARQUET_FOLDER', 'parquets')
        metadata_path = os.path.join(parquet_folder, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                for key, value in metadata.items():
                    if isinstance(value, list) and len(value) > 0:
                        descriptions[key] = value[0].get('description', key)
                    else:
                        descriptions[key] = key
        else:
            # Fallback: try different common locations
            possible_paths = [
                'parquets/POCO_MRO_003/metadata.json',
                'parquets/metadata.json',
                'metadata.json'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        for key, value in metadata.items():
                            if isinstance(value, list) and len(value) > 0:
                                descriptions[key] = value[0].get('description', key)
                            else:
                                descriptions[key] = key
                    break
    except Exception as e:
        print(f"Warning: Could not load metadata descriptions: {e}")
        # Create fallback descriptions using the series names
        descriptions = {}

    return descriptions

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

    # Calculate global date range for slider
    global_min_date = None
    global_max_date = None
    if detector.dataframes:
        for df in detector.dataframes.values():
            if len(df) > 0:
                series_min = df.index.min()
                series_max = df.index.max()
                if global_min_date is None or series_min < global_min_date:
                    global_min_date = series_min
                if global_max_date is None or series_max > global_max_date:
                    global_max_date = series_max

    return detector, series_options, global_min_date, global_max_date

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
detector, series_options, global_min_date, global_max_date = initialize_detector()

# Load metadata descriptions
metadata_descriptions = load_metadata_descriptions()

# Global variables to be updated when reloading data
global_detector = detector
global_series_options = series_options
global_min_date = global_min_date
global_max_date = global_max_date

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
        selected_series = [global_series_options[0]['value']] if global_series_options else []
    if selected_methods is None:
        selected_methods = ['IF']

    # Pre-compute initial figure so each panel renders without interaction
    if selected_series and global_detector:
        initial_figure = global_detector.plot_multiple_series(selected_series, 'Value', selected_methods)
    else:
        initial_figure = go.Figure()
        initial_figure.update_layout(
            title="Select at least one time series",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )

    # Create checklist options with tooltips
    checklist_items = []
    tooltips = []

    for option in global_series_options:
        # Create a unique ID for each checkbox item
        item_id = f"checkbox-{panel_id}-{option['value']}"

        # Create the checkbox item with custom styling
        item = html.Div([
            html.Div([
                dcc.Checklist(
                    id={'type': 'individual-checkbox', 'panel': panel_id, 'series': option['value']},
                    options=[{'label': option['label'], 'value': option['value']}],
                    value=[option['value']] if option['value'] in selected_series else [],
                    labelStyle={'display': 'block', 'margin-bottom': '5px', 'fontSize': '12px', 'cursor': 'pointer'},
                    inputStyle={"margin-right": "5px"},
                    style={"display": "inline-block"}
                )
            ], id=item_id, style={"width": "100%"})
        ], style={"width": "100%", "margin-bottom": "2px"})

        checklist_items.append(item)

        # Add tooltip for this item
        description = metadata_descriptions.get(option['value'], option['label'])
        tooltip = dbc.Tooltip(
            description,
            target=item_id,
            placement="right",
            delay={"show": 500, "hide": 100}
        )
        tooltips.append(tooltip)

    return dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H6("Time Series", className="fw-bold"),
                                    html.Div(
                                        checklist_items,
                                        style={"maxHeight": "220px", "overflowY": "auto", "width": "100%"}
                                    ),
                                    # Hidden main checklist to maintain callback compatibility
                                    dcc.Checklist(
                                        id={'type': 'series-selector-checklist', 'index': panel_id},
                                        options=global_series_options,
                                        value=selected_series,
                                        style={"display": "none"}
                                    ),
                                    # Add all tooltips
                                    *[tooltip for tooltip in tooltips],
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
default_series = [global_series_options[0]['value']] if global_series_options else []

def configure_global_slider(min_date, max_date):
    """Configure slider parameters based on date range."""
    global_slider_marks = {}
    if min_date and max_date:
        global_slider_min = min_date.timestamp()
        global_slider_max = max_date.timestamp()
        global_slider_value = [global_slider_min, global_slider_max]
        # Add marks for start and end dates, plus intermediate
        import pandas as pd
        global_slider_marks = {global_slider_min: min_date.strftime('%Y-%m-%d'), global_slider_max: max_date.strftime('%Y-%m-%d')}
        # Add monthly marks
        date_range = pd.date_range(start=min_date, end=max_date, freq='M')
        for d in date_range:
            if d != min_date and d != max_date:
                global_slider_marks[d.timestamp()] = d.strftime('%b %Y')
    else:
        global_slider_min = 0
        global_slider_max = 1
        global_slider_value = [0, 1]

    return global_slider_min, global_slider_max, global_slider_value, global_slider_marks

# Configure initial global slider
global_slider_min, global_slider_max, global_slider_value, global_slider_marks = configure_global_slider(global_min_date, global_max_date)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("FIELD LABEL", className="text-center my-4", style={"color": "#2c3e50"}),
            html.Hr()
        ], width=12)
    ]),
    # Parquet folder input with apply button
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Parquet Folder:"),
                dbc.Input(
                    id='parquet-folder-input',
                    type='text',
                    value=os.getenv('PARQUET_FOLDER', 'parquets'),
                    placeholder="Enter parquet folder path"
                ),
                dbc.Button("Apply", id='apply-folder-button', color="primary", className="ms-2")
            ]),
        ], width=12, className="mb-3")
    ]),
    # Global time range slider
    dbc.Row([
        dbc.Col([
            html.H5("Global Time Range", className="fw-bold"),
            dcc.RangeSlider(
                id='global-range-slider',
                min=global_slider_min,
                max=global_slider_max,
                value=global_slider_value,
                marks=global_slider_marks,
                step=86400,  # 1 day in seconds
                allowCross=False,
                tooltip={"placement": "bottom"}
            )
        ], width=12, className="mb-4")
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

# Callback to reload data from new parquet folder
@app.callback(
    [Output('global-range-slider', 'min'),
     Output('global-range-slider', 'max'),
     Output('global-range-slider', 'value'),
     Output('global-range-slider', 'marks'),
     Output({'type': 'series-selector-checklist', 'index': ALL}, 'options'),
     Output({'type': 'series-selector-checklist', 'index': ALL}, 'value')],
    [Input('apply-folder-button', 'n_clicks')],
    [State('parquet-folder-input', 'value'),
     State({'type': 'series-selector-checklist', 'index': ALL}, 'id'),
     State({'type': 'series-selector-checklist', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def reload_data_from_folder(n_clicks, parquet_folder, checklist_ids, checklist_values):
    """Reload data from the specified parquet folder."""
    global global_detector, global_series_options, global_min_date, global_max_date, metadata_descriptions

    if not parquet_folder or parquet_folder.strip() == "":
        # If empty, reload with default
        parquet_folder = os.getenv('PARQUET_FOLDER', 'parquets')

    try:
        # Temporarily override environment variable
        original_parquet_folder = os.environ.get('PARQUET_FOLDER')
        os.environ['PARQUET_FOLDER'] = parquet_folder.strip()

        # Reload data
        new_detector, new_series_options, new_min_date, new_max_date = initialize_detector()

        # Update global variables
        global_detector = new_detector
        global_series_options = new_series_options
        global_min_date = new_min_date
        global_max_date = new_max_date

        # Configure new slider parameters
        slider_min, slider_max, slider_value, slider_marks = configure_global_slider(new_min_date, new_max_date)

        # Update series options for all checklists
        series_options_list = [new_series_options] * len(checklist_ids)

        # Update selected values to ensure they exist in new options
        updated_values = []
        for values in checklist_values:
            if values:
                # Keep only values that exist in new options
                valid_values = [v for v in values if any(opt['value'] == v for opt in new_series_options)]
                if not valid_values and new_series_options:
                    # If no valid values, select first available
                    valid_values = [new_series_options[0]['value']]
                updated_values.append(valid_values)
            else:
                # If no values selected, select first available
                updated_values.append([new_series_options[0]['value']] if new_series_options else [])

        # Restore original environment variable
        if original_parquet_folder is not None:
            os.environ['PARQUET_FOLDER'] = original_parquet_folder
        elif 'PARQUET_FOLDER' in os.environ:
            del os.environ['PARQUET_FOLDER']

        return slider_min, slider_max, slider_value, slider_marks, series_options_list, updated_values

    except Exception as e:
        print(f"Error reloading data: {e}")
        # Return no updates on error
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback to handle individual checkbox changes
@app.callback(
    Output({'type': 'series-selector-checklist', 'index': ALL}, 'value', allow_duplicate=True),
    [Input({'type': 'individual-checkbox', 'panel': ALL, 'series': ALL}, 'value')],
    [State({'type': 'individual-checkbox', 'panel': ALL, 'series': ALL}, 'id'),
     State({'type': 'series-selector-checklist', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_series_selector_from_individual_checkboxes(individual_values, individual_ids, checklist_ids):
    """Update the main series selectors when individual checkboxes change."""
    # Group by panel
    panel_updates = {}

    # Initialize all panels
    for checklist_id in checklist_ids:
        panel_id = checklist_id['index']
        panel_updates[panel_id] = set()

    # Collect selected series per panel
    for i, checkbox_id in enumerate(individual_ids):
        panel_id = checkbox_id['panel']
        series_name = checkbox_id['series']
        is_checked = len(individual_values[i]) > 0

        if is_checked:
            if panel_id in panel_updates:
                panel_updates[panel_id].add(series_name)

    # Return updates in the same order as checklist_ids
    result = []
    for checklist_id in checklist_ids:
        panel_id = checklist_id['index']
        result.append(list(panel_updates.get(panel_id, set())))

    return result

# Callback to update panel selections in store
@app.callback(
    [Output('graph-container', 'children'),
     Output('graph-store', 'data', allow_duplicate=True),
     Output('global-range-slider', 'min', allow_duplicate=True),
     Output('global-range-slider', 'max', allow_duplicate=True),
     Output('global-range-slider', 'value', allow_duplicate=True),
     Output('global-range-slider', 'marks', allow_duplicate=True)],
    [Input('add-graph-button', 'n_clicks'),
     Input({'type': 'delete-graph-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'series-selector-checklist', 'index': ALL}, 'value'),
     Input({'type': 'methods-selector-checklist', 'index': ALL}, 'value')],
    [State('graph-store', 'data'),
     State('global-range-slider', 'value')],
    prevent_initial_call=True
)
def manage_graph_panels(add_clicks, delete_clicks, series_values, methods_values, current_panels, current_slider_value):
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

    # Calculate new slider range based on currently selected series across all panels
    active_min_date = None
    active_max_date = None

    # Collect all selected series across all panels
    all_selected_series = set()
    for idx, panel in enumerate(current_panels):
        panel_series = series_values[idx] if idx < len(series_values) else panel.get('series', [])
        all_selected_series.update(panel_series)

    # Calculate range based only on selected series
    for series_name in all_selected_series:
        if series_name in global_detector.dataframes:
            df = global_detector.dataframes[series_name]
            if len(df) > 0:
                series_min = df.index.min()
                series_max = df.index.max()
                if active_min_date is None or series_min < active_min_date:
                    active_min_date = series_min
                if active_max_date is None or series_max > active_max_date:
                    active_max_date = series_max

    # Configure slider based on active series range
    if active_min_date and active_max_date:
        slider_min, slider_max, slider_value, slider_marks = configure_global_slider(active_min_date, active_max_date)
        # Preserve current slider position if it's within the new range
        if current_slider_value and len(current_slider_value) == 2:
            preserved_value = [
                max(slider_min, min(current_slider_value[0], slider_max)),
                min(slider_max, max(current_slider_value[1], slider_min))
            ]
            slider_value = preserved_value
    else:
        # No series selected, use minimal range
        slider_min = 0
        slider_max = 1
        slider_value = [0, 1]
        slider_marks = {}

    return panel_components, current_panels, slider_min, slider_max, slider_value, slider_marks

@app.callback(
    Output({'type': 'anomalies-graph', 'index': MATCH}, 'figure'),
    [Input({'type': 'series-selector-checklist', 'index': MATCH}, 'value'),
     Input({'type': 'methods-selector-checklist', 'index': MATCH}, 'value'),
     Input('global-range-slider', 'value')],
    [State({'type': 'anomalies-graph', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_individual_graph(selected_series, selected_methods, global_slider_value, graph_id):
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

    # Convert global slider values to datetime
    start_date = None
    end_date = None
    if global_slider_value and len(global_slider_value) == 2:
        start_date = datetime.fromtimestamp(global_slider_value[0])
        end_date = datetime.fromtimestamp(global_slider_value[1])

    # Create graph for this specific panel
    return global_detector.plot_multiple_series(selected_series, 'Value', selected_methods, start_date, end_date)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
