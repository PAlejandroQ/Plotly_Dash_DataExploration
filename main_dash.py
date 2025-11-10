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

def get_directory_structure(max_depth=2):
    """
    Get directory structure based on PARQUET_FOLDER environment variable.
    Shows only directories up to max_depth levels.
    Returns a formatted string showing the directory tree.
    """
    base_path = os.getenv('PARQUET_FOLDER', 'parquets')

    if not os.path.exists(base_path):
        return f"❌ Directory '{base_path}' does not exist."

    if not os.path.isdir(base_path):
        return f"❌ '{base_path}' is not a directory."

    def explore_dir(path, current_depth=0, prefix=""):
        if current_depth > max_depth:
            return []

        lines = []
        try:
            items = os.listdir(path)
            dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
            dirs.sort()

            if not dirs:
                return lines

            for i, dir_name in enumerate(dirs):
                full_path = os.path.join(path, dir_name)
                is_last = (i == len(dirs) - 1)

                # Add current directory
                if current_depth == 0:
                    lines.append(f"📁 {full_path}/")
                else:
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}📁 {full_path}/")

                # Explore subdirectories if not at max depth
                if current_depth < max_depth:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    sub_lines = explore_dir(full_path, current_depth + 1, next_prefix)
                    lines.extend(sub_lines)

        except PermissionError:
            lines.append(f"{prefix}└── 📁 [Permission denied]")
        except Exception as e:
            lines.append(f"{prefix}└── [Error: {str(e)}]")

        return lines

    lines = explore_dir(base_path)
    if not lines:
        return f"📂 No subdirectories found in '{base_path}'."

    return "\n".join(lines)

# Load metadata descriptions
def load_metadata_descriptions(parquet_folder=None):
    """Load descriptions from metadata.json file."""
    descriptions = {}
    metadata_message = ""

    if parquet_folder is None:
        return descriptions, "No parquet folder specified for metadata loading."

    try:
        metadata_path = os.path.join(parquet_folder, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                for key, value in metadata.items():
                    if isinstance(value, list) and len(value) > 0:
                        descriptions[key] = value[0].get('description', key)
                    else:
                        descriptions[key] = key
            metadata_message = f"Loaded metadata descriptions from {metadata_path}."
        else:
            metadata_message = f"Metadata file not found at {metadata_path}. Using series names as descriptions."
    except Exception as e:
        metadata_message = f"Could not load metadata descriptions: {str(e)}. Using series names as descriptions."
        descriptions = {}

    return descriptions, metadata_message

def initialize_detector(parquet_folder=None):
    """
    Initializes the detector by loading data from Parquet files only.
    If parquet_folder is None, returns empty initialization.
    """
    detector = TimeSeriesAnomalyDetector()

    # If no parquet folder provided, return empty initialization
    if parquet_folder is None:
        return detector, [], None, None

    series_options = []
    success_message = ""
    error_message = ""

    print(f"Loading data from Parquet files in: {parquet_folder}")
    try:
        series_options = detector.load_series_from_parquet(parquet_folder, target_col='Value')
        if series_options:
            print(f"Successfully loaded {len(series_options)} series from Parquet files")
            success_message = f"Loaded {len(series_options)} time series from Parquet files."
        else:
            error_message = "No Parquet files found in the specified folder."
    except Exception as e:
        error_message = f"Error loading Parquet files: {str(e)}"

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

    return detector, series_options, global_min_date, global_max_date, success_message, error_message

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

# Initialize with empty data - will be loaded when user clicks Apply
global_detector = TimeSeriesAnomalyDetector()
global_series_options = []
global_min_date = None
global_max_date = None
metadata_descriptions = {}
global_messages = {"success": "", "error": "", "metadata": ""}

def create_graph_panel(panel_id, selected_series=None):
    """
    Creates a template for a dynamic graph panel.

    Args:
        panel_id: Unique ID for the panel
        selected_series: List of selected series (optional)

    Returns:
        A dbc.Row component with the complete panel
    """
    # Default values if not provided
    if selected_series is None:
        selected_series = [global_series_options[0]['value']] if global_series_options else []

    # Pre-compute initial figure so each panel renders without interaction
    if selected_series and global_detector and global_detector.dataframes:
        initial_figure = global_detector.plot_multiple_series(selected_series, 'Value')
    else:
        initial_figure = go.Figure()
        initial_figure.update_layout(
            title="No data loaded yet. Click 'Apply' to load data from parquet folder.",
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
                                        style={"maxHeight": "470px", "overflowY": "auto", "width": "100%"}
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
                        ], width=9)
                    ], className="g-3", align="stretch")
                ]),
                className="mb-4",
                style={"height": "100%", "overflow": "hidden"}
            )
        ], width=12)
    ])

# Initial layout with one default panel
initial_panel_id = str(uuid.uuid4())[:8]
default_series = []

# Store to manage panel states (series selections only)
initial_panel_data = [{
    'id': initial_panel_id,
    'series': default_series
}]

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
    # Directory structure display
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("Available Directories", className="mb-0"),
                    html.Small("Directories found in PARQUET_FOLDER. Select and copy any path to paste in the input above.", className="text-muted d-block")
                ]),
                dbc.CardBody([
                    html.Pre(
                        get_directory_structure(),
                        id='directory-structure-display',
                        style={
                            'font-family': 'monospace',
                            'font-size': '12px',
                            'line-height': '1.4',
                            'margin': '0',
                            'white-space': 'pre-wrap',
                            'word-wrap': 'break-word',
                            'max-height': '300px',
                            'overflow-y': 'auto',
                            'cursor': 'pointer',
                            'user-select': 'text'
                        }
                    )
                ])
            ])
        ], width=12, className="mb-4")
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

    # Store to manage panel states (series selections only)
    dcc.Store(id='graph-store', data=initial_panel_data),

    # Dynamic graph container
    html.Div(
        id='graph-container',
        children=[create_graph_panel(initial_panel_id, default_series)],
        style={"width": "100%"}
    ),

    # Messages display area
    html.Div(
        id='messages-container',
        children=[],
        style={"marginBottom": "20px"}
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
     Output({'type': 'series-selector-checklist', 'index': ALL}, 'value'),
     Output('graph-container', 'children', allow_duplicate=True),
     Output('messages-container', 'children')],
    [Input('apply-folder-button', 'n_clicks')],
    [State('parquet-folder-input', 'value'),
     State({'type': 'series-selector-checklist', 'index': ALL}, 'id'),
     State({'type': 'series-selector-checklist', 'index': ALL}, 'value'),
     State('graph-store', 'data')],
    prevent_initial_call=True
)
def reload_data_from_folder(n_clicks, parquet_folder, checklist_ids, checklist_values, current_panels):
    """Reload data from the specified parquet folder."""
    global global_detector, global_series_options, global_min_date, global_max_date, metadata_descriptions, global_messages

    if not parquet_folder or parquet_folder.strip() == "":
        # If empty, reload with default
        parquet_folder = os.getenv('PARQUET_FOLDER', 'parquets')

    try:
        # Temporarily override environment variable
        original_parquet_folder = os.environ.get('PARQUET_FOLDER')
        os.environ['PARQUET_FOLDER'] = parquet_folder.strip()

        # Load metadata descriptions
        metadata_descriptions, metadata_message = load_metadata_descriptions(parquet_folder.strip())

        # Reload data
        new_detector, new_series_options, new_min_date, new_max_date, success_message, error_message = initialize_detector(parquet_folder.strip())

        # Update global variables
        global_detector = new_detector
        global_series_options = new_series_options
        global_min_date = new_min_date
        global_max_date = new_max_date

        # Prepare messages for display
        messages = []
        if success_message:
            messages.append(dbc.Alert(success_message, color="success", dismissable=True))
        if error_message:
            messages.append(dbc.Alert(error_message, color="warning", dismissable=True))
        if metadata_message:
            messages.append(dbc.Alert(metadata_message, color="info", dismissable=True))

        global_messages = {
            "success": success_message,
            "error": error_message,
            "metadata": metadata_message
        }

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

        # Update panels with new data
        panel_components = []
        if current_panels:
            for panel in current_panels:
                panel_id = panel['id']
                selected_series = panel.get('series', [])
                # Filter to only include valid series
                valid_series = [s for s in selected_series if any(opt['value'] == s for opt in new_series_options)]
                if not valid_series and new_series_options:
                    valid_series = [new_series_options[0]['value']]
                panel_components.append(create_graph_panel(panel_id, valid_series))
        else:
            # Create default panel if none exist
            panel_id = str(uuid.uuid4())[:8]
            default_series = [new_series_options[0]['value']] if new_series_options else []
            panel_components = [create_graph_panel(panel_id, default_series)]

        # Restore original environment variable
        if original_parquet_folder is not None:
            os.environ['PARQUET_FOLDER'] = original_parquet_folder
        elif 'PARQUET_FOLDER' in os.environ:
            del os.environ['PARQUET_FOLDER']

        return slider_min, slider_max, slider_value, slider_marks, series_options_list, updated_values, panel_components, messages

    except Exception as e:
        print(f"Error reloading data: {e}")
        error_alert = dbc.Alert(f"Unexpected error: {str(e)}", color="danger", dismissable=True)
        # Return no updates on error
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, [error_alert]

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
     Input({'type': 'series-selector-checklist', 'index': ALL}, 'value')],
    [State('graph-store', 'data'),
     State('global-range-slider', 'value')],
    prevent_initial_call=True
)
def manage_graph_panels(add_clicks, delete_clicks, series_values, current_panels, current_slider_value):
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

    # Update stored selections for current panels
    for idx, panel in enumerate(current_panels):
        if idx < len(series_values):
            panel['series'] = series_values[idx] or []

    # Determine which action was performed
    if trigger_id == 'add-graph-button':
        new_panel_id = str(uuid.uuid4())[:8]
        current_panels.append({
            'id': new_panel_id,
            'series': default_series.copy()
        })
    elif isinstance(trigger_id, dict) and trigger_id.get('type') == 'delete-graph-button':
        delete_index = trigger_id['index']
        current_panels = [panel for panel in current_panels if panel['id'] != delete_index]

    # Ensure at least one panel exists
    if not current_panels:
        new_panel_id = str(uuid.uuid4())[:8]
        current_panels = [{
            'id': new_panel_id,
            'series': default_series.copy()
        }]

    # Create components for all panels using their current state
    panel_components = []
    for panel in current_panels:
        panel_id = panel['id']
        selected_series = panel.get('series', []) or default_series.copy()
        panel_components.append(create_graph_panel(panel_id, selected_series))

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
     Input('global-range-slider', 'value')],
    [State({'type': 'anomalies-graph', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_individual_graph(selected_series, global_slider_value, graph_id):
    """
    Callback to update individual graphs using MATCH pattern.
    """
    panel_id = graph_id['index']

    if not selected_series or not global_detector.dataframes:
        # If no series are selected or no data is loaded, show appropriate message
        import plotly.graph_objects as go
        fig = go.Figure()
        if not global_detector.dataframes:
            title = "No data loaded yet. Click 'Apply' to load data from parquet folder."
        else:
            title = "Select at least one time series"
        fig.update_layout(
            title=title,
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
    return global_detector.plot_multiple_series(selected_series, 'Value', start_date, end_date)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
