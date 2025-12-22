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
    Shows only directories up to max_depth levels with absolute paths.
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
                abs_path = os.path.abspath(full_path)
                is_last = (i == len(dirs) - 1)

                # Add current directory with absolute path
                if current_depth == 0:
                    lines.append(f"📁 {abs_path}/")
                else:
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}📁 {abs_path}/")

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

# Load metadata descriptions and units with filtering
def load_metadata_descriptions(parquet_folder=None):
    """Load descriptions and units from metadata.json file, filtering by allowed units."""
    descriptions = {}
    units = {}
    metadata_message = ""

    # Allowed units for multiple Y-axes
    allowed_units = {'kgf/cm² a', 'ºC', 'l', 'm3/d'}

    if parquet_folder is None:
        return descriptions, units, "No parquet folder specified for metadata loading."

    try:
        # Try multiple possible locations for metadata.json
        possible_paths = [
            os.path.join(parquet_folder, 'metadata.json'),
            os.path.join(parquet_folder, 'POCO_MRO_003', 'metadata.json'),
            'parquets/POCO_MRO_003/metadata.json'
        ]

        metadata_path = None
        for path in possible_paths:
            if os.path.exists(path):
                metadata_path = path
                break

        if metadata_path:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                filtered_count = 0
                total_count = 0

                for key, value in metadata.items():
                    total_count += 1
                    if isinstance(value, list) and len(value) > 0:
                        unit = value[0].get('unit', '')
                        if unit in allowed_units:
                            descriptions[key] = value[0].get('description', key)
                            units[key] = unit
                            filtered_count += 1
                        # Skip series with units not in allowed_units

                metadata_message = f"Loaded metadata from {metadata_path}. Filtered {filtered_count}/{total_count} series with allowed units: {', '.join(allowed_units)}."
        else:
            metadata_message = f"Metadata file not found at {metadata_path}. Using series names as descriptions."
    except Exception as e:
        metadata_message = f"Could not load metadata descriptions: {str(e)}. Using series names as descriptions."
        descriptions = {}
        units = {}

    return descriptions, units, metadata_message

# Load anomaly events from events.json file
def load_anomaly_events(parquet_folder=None):
    """Load anomaly events from events.json file containing timestamp ranges."""
    events = []
    events_message = ""

    if parquet_folder is None:
        return events, "No parquet folder specified for events loading."

    try:
        # Try multiple possible locations for events.json
        possible_paths = [
            os.path.join(parquet_folder, 'events.json'),
            os.path.join(parquet_folder, 'POCO_MRO_003', 'events.json'),
            'parquets/POCO_MRO_003/events.json'
        ]

        events_path = None
        for path in possible_paths:
            if os.path.exists(path):
                events_path = path
                break

        if events_path:
            with open(events_path, 'r', encoding='utf-8') as f:
                events_data = json.load(f)
                # events.json contains a list of [start, end] timestamp pairs
                if isinstance(events_data, list):
                    events = events_data
                    events_message = f"Loaded {len(events)} base anomaly events from {events_path}."
                else:
                    events_message = f"Invalid format in events.json: expected list of timestamp pairs."
        else:
            events_message = f"Events file not found at {events_path}. No base anomaly highlighting will be applied."
    except Exception as e:
        events_message = f"Could not load base anomaly events: {str(e)}. No base anomaly highlighting will be applied."
        events = []

    return events, events_message

# Load vigres events from events_from_vigres.json file
def load_vigres_events(parquet_folder=None):
    """Load vigres events from events_from_vigres.json file containing timestamp ranges."""
    vigres_events = []
    vigres_message = ""

    if parquet_folder is None:
        return vigres_events, "No parquet folder specified for vigres events loading."

    try:
        # Try multiple possible locations for events_from_vigres.json
        possible_paths = [
            os.path.join(parquet_folder, 'events_from_vigres.json'),
            os.path.join(parquet_folder, 'POCO_MRO_003', 'events_from_vigres.json'),
            'parquets/POCO_MRO_003/events_from_vigres.json'
        ]

        vigres_path = None
        for path in possible_paths:
            if os.path.exists(path):
                vigres_path = path
                break

        if vigres_path:
            with open(vigres_path, 'r', encoding='utf-8') as f:
                vigres_data = json.load(f)
                # events_from_vigres.json contains a list of [start, end] timestamp pairs
                if isinstance(vigres_data, list):
                    vigres_events = vigres_data
                    vigres_message = f"Loaded {len(vigres_events)} vigres events from {vigres_path}."
                else:
                    vigres_message = f"Invalid format in events_from_vigres.json: expected list of timestamp pairs."
        else:
            vigres_message = f"Vigres events file not found at {vigres_path}. No vigres highlighting will be applied."
    except Exception as e:
        vigres_message = f"Could not load vigres events: {str(e)}. No vigres highlighting will be applied."
        vigres_events = []

    return vigres_events, vigres_message

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
metadata_units = {}
global_anomaly_events = []
global_messages = {"success": "", "error": "", "metadata": "", "events": ""}

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
        initial_figure, _ = global_detector.plot_multiple_series(selected_series, 'Value', units_dict=metadata_units, anomaly_events=global_anomaly_events, vigres_events=global_vigres_events)
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
                                    html.H6("Events", className="fw-bold"),
                                    html.Div(
                                        id={'type': 'events-buttons-container', 'index': panel_id},
                                        style={"maxHeight": "200px", "overflowY": "auto", "width": "100%"}
                                    ),
                                    html.Hr(className="my-3"),
                                    html.Div([
                                        dbc.Button(
                                            "Reset axes",
                                            id={'type': 'reset-axes-button', 'index': panel_id},
                                            color="secondary",
                                            size="sm",
                                            style={"fontSize": "11px", "marginRight": "5px", "padding": "2px 6px"}
                                        ),
                                        dbc.Button(
                                            "×",
                                            id={'type': 'delete-graph-button', 'index': panel_id},
                                            color="danger",
                                            size="sm",
                                            style={"fontSize": "18px", "padding": "0 8px"}
                                        ),
                                    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                                    # Store for events data
                                    dcc.Store(id={'type': 'events-store', 'index': panel_id}, data=[])
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

        # Load metadata descriptions and units
        global metadata_descriptions, metadata_units
        metadata_descriptions, metadata_units, metadata_message = load_metadata_descriptions(parquet_folder.strip())

        # Load anomaly events
        global global_anomaly_events
        global_anomaly_events, events_message = load_anomaly_events(parquet_folder.strip())

        # Load vigres events
        global global_vigres_events
        global_vigres_events, vigres_message = load_vigres_events(parquet_folder.strip())

        # Reload data
        new_detector, new_series_options, new_min_date, new_max_date, success_message, error_message = initialize_detector(parquet_folder.strip())

        # Filter series options to only include allowed units
        allowed_units = {'kgf/cm² a', 'ºC', 'l', 'm3/d'}
        filtered_series_options = [
            option for option in new_series_options
            if option['value'] in metadata_units and metadata_units[option['value']] in allowed_units
        ]

        # Update global variables
        global_detector = new_detector
        global_series_options = filtered_series_options
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
        if events_message:
            messages.append(dbc.Alert(events_message, color="info", dismissable=True))
        if vigres_message:
            messages.append(dbc.Alert(vigres_message, color="info", dismissable=True))

        global_messages = {
            "success": success_message,
            "error": error_message,
            "metadata": metadata_message,
            "events": events_message,
            "vigres": vigres_message
        }

        # Configure new slider parameters
        slider_min, slider_max, slider_value, slider_marks = configure_global_slider(new_min_date, new_max_date)

        # Update series options for all checklists (filtered)
        # Ensure we return a list with the correct number of elements for ALL output
        series_options_list = [filtered_series_options] * len(checklist_ids) if checklist_ids else []

        # Update selected values to ensure they exist in new options
        # Ensure we return a list with the correct number of elements for ALL output
        updated_values = []
        for values in checklist_values:
            if values:
                # Keep only values that exist in filtered options
                valid_values = [v for v in values if any(opt['value'] == v for opt in filtered_series_options)]
                if not valid_values and filtered_series_options:
                    # If no valid values, select first available
                    valid_values = [filtered_series_options[0]['value']] if filtered_series_options else []
                updated_values.append(valid_values)
            else:
                # If no values selected, select first available
                updated_values.append([filtered_series_options[0]['value']] if filtered_series_options else [])

        # Ensure the lists have the correct size for ALL outputs
        if len(updated_values) != len(checklist_ids):
            # Pad or truncate to match checklist_ids length
            while len(updated_values) < len(checklist_ids):
                updated_values.append([filtered_series_options[0]['value']] if filtered_series_options else [])
            if len(updated_values) > len(checklist_ids):
                updated_values = updated_values[:len(checklist_ids)]

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
                # Create panel with anomaly events
                panel_components.append(create_graph_panel(panel_id, valid_series))
        else:
            # Create default panel if none exist
            panel_id = str(uuid.uuid4())[:8]
            default_series = [filtered_series_options[0]['value']] if filtered_series_options else []
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
        # Return appropriate values on error - cannot use dash.no_update for ALL outputs
        # Return empty lists for checklist options and values, and no changes for others
        empty_options = [] if checklist_ids else []
        empty_values = [] if checklist_ids else []
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, empty_options, empty_values, dash.no_update, [error_alert]

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
     Input({'type': 'reset-axes-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'series-selector-checklist', 'index': ALL}, 'value')],
    [State('graph-store', 'data'),
     State('global-range-slider', 'value')],
    prevent_initial_call=True
)
def manage_graph_panels(add_clicks, delete_clicks, reset_clicks, series_values, current_panels, current_slider_value):
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
    reset_clicks = reset_clicks or []

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
    elif isinstance(trigger_id, dict) and trigger_id.get('type') == 'reset-axes-button':
        # Reset axes for the specific panel - clear zoom event
        reset_panel_id = trigger_id['index']
        for panel in current_panels:
            if panel['id'] == reset_panel_id:
                if 'zoom_event' in panel:
                    del panel['zoom_event']
                break

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
    [Input('graph-store', 'data'),
     Input({'type': 'series-selector-checklist', 'index': MATCH}, 'value'),
     Input('global-range-slider', 'value')],
    [State({'type': 'anomalies-graph', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_graph_with_zoom(store_data, selected_series, global_slider_value, graph_id):
    """
    Update graph with zoom if zoom_event exists in panel data.
    """
    panel_id = graph_id['index']
    print(f"📊 Updating graph for panel {panel_id}")

    # Find panel data
    panel_data = None
    if store_data:
        for panel in store_data:
            if panel['id'] == panel_id:
                panel_data = panel
                break

    if not panel_data:
        print(f"❌ Panel {panel_id} not found in store")
        return dash.no_update

    # Check if there's a zoom event to apply
    zoom_event = panel_data.get('zoom_event')
    if zoom_event:
        print(f"🔍 Applying zoom for panel {panel_id}: {zoom_event['start']} - {zoom_event['end']}")

        # Use zoom range instead of global slider range
        start_date = pd.to_datetime(zoom_event['start'])
        end_date = pd.to_datetime(zoom_event['end'])

        # Check if zoom event is recent (within last 2 seconds)
        zoom_timestamp = pd.to_datetime(zoom_event.get('timestamp', '2000-01-01'))
        time_diff = pd.Timestamp.now() - zoom_timestamp

        if time_diff.total_seconds() > 2:
            print("⏰ Zoom event expired, using global range")
            panel_data['zoom_event'] = None
            # Use global slider range
            if global_slider_value and len(global_slider_value) == 2:
                start_date = datetime.fromtimestamp(global_slider_value[0])
                end_date = datetime.fromtimestamp(global_slider_value[1])
            else:
                start_date = None
                end_date = None
        else:
            print("✅ Using zoom range for graph update")
    else:
        # Use global slider range
        if global_slider_value and len(global_slider_value) == 2:
            start_date = datetime.fromtimestamp(global_slider_value[0])
            end_date = datetime.fromtimestamp(global_slider_value[1])
        else:
            start_date = None
            end_date = None

    if not selected_series or not global_detector.dataframes:
        # Return empty figure
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="No data loaded yet. Click 'Apply' to load data from parquet folder.",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )
        return fig

    # Create graph with appropriate range
    fig, visible_events = global_detector.plot_multiple_series(
        selected_series, 'Value', start_date, end_date,
        metadata_units, global_anomaly_events, vigres_events=global_vigres_events
    )

    print(f"📈 Graph updated for panel {panel_id}")
    return fig

@app.callback(
    [Output({'type': 'events-buttons-container', 'index': MATCH}, 'children'),
     Output({'type': 'events-store', 'index': MATCH}, 'data')],
    [Input({'type': 'series-selector-checklist', 'index': MATCH}, 'value'),
     Input('global-range-slider', 'value')],
    [State({'type': 'events-buttons-container', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_events_buttons(selected_series, global_slider_value, container_id):
    """
    Callback to update the events buttons container.
    """
    panel_id = container_id['index']

    if not selected_series or not global_detector.dataframes:
        return html.Div("No events available", style={"fontSize": "12px", "color": "#666"})

    # Convert global slider values to datetime
    start_date = None
    end_date = None
    if global_slider_value and len(global_slider_value) == 2:
        start_date = datetime.fromtimestamp(global_slider_value[0])
        end_date = datetime.fromtimestamp(global_slider_value[1])

    # Get visible events
    _, visible_events = global_detector.plot_multiple_series(selected_series, 'Value', start_date, end_date, metadata_units, global_anomaly_events, vigres_events=global_vigres_events)

    if not visible_events:
        return html.Div("No events in current view", style={"fontSize": "12px", "color": "#666"}), visible_events

    # Create buttons for each visible event
    buttons = []
    for i, event in enumerate(visible_events):
        color = "danger" if event['type'] == 'anomaly' else "success"
        buttons.append(
            dbc.Button(
                event['name'],
                id={'type': 'event-focus-button', 'index': f"{panel_id}_{i}"},
                color=color,
                size="sm",
                className="mb-1 me-1",
                style={"fontSize": "11px", "padding": "2px 6px", "width": "100%"},
                n_clicks=0
            )
        )

    return html.Div(buttons, style={"display": "flex", "flexDirection": "column"}), visible_events

@app.callback(
    Output('graph-store', 'data', allow_duplicate=True),
    Input({'type': 'reset-axes-button', 'index': ALL}, 'n_clicks'),
    [State('graph-store', 'data'),
     State({'type': 'reset-axes-button', 'index': ALL}, 'id'),
     State('global-range-slider', 'value')],
    prevent_initial_call=True
)
def handle_reset_axes(n_clicks_list, graph_store_data, button_ids, global_slider_value):
    """
    Handle reset axes button clicks by clearing any zoom events and applying the global slider range.
    """
    # Check if any reset button was actually clicked
    if not n_clicks_list or all(clicks == 0 for clicks in n_clicks_list):
        return dash.no_update

    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]['prop_id']
    import json
    try:
        button_info = json.loads(triggered_id.split('.')[0])
        panel_id = button_info['index']

        print(f"🔄 RESET AXES: Panel {panel_id}")

        # Find the panel in graph store
        if graph_store_data:
            for panel in graph_store_data:
                if isinstance(panel, dict) and panel.get('id') == panel_id:
                    # Clear any zoom event
                    if 'zoom_event' in panel:
                        del panel['zoom_event']
                        print(f"🗑️ Cleared zoom event for panel {panel_id}")
                    break

        return graph_store_data

    except Exception as e:
        print(f"❌ Error in handle_reset_axes: {e}")
        return dash.no_update

@app.callback(
    Output('graph-store', 'data', allow_duplicate=True),
    Input({'type': 'event-focus-button', 'index': ALL}, 'n_clicks'),
    [State('graph-store', 'data'),
     State({'type': 'event-focus-button', 'index': ALL}, 'id'),
     State({'type': 'events-store', 'index': ALL}, 'data'),
     State({'type': 'events-store', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def handle_event_zoom(n_clicks_list, graph_store_data, button_ids, events_data_list, events_store_ids):
    """
    Handle event focus button clicks by updating the store with zoom information.
    This will trigger the graph updates through the existing update_individual_graph callback.
    """
    # Check if any button was actually clicked (n_clicks > 0)
    if not n_clicks_list or all(clicks == 0 for clicks in n_clicks_list):
        # No button was clicked, return no update
        return dash.no_update
    """
    Handle event focus button clicks by updating the store with zoom information.
    This will trigger the graph updates through the existing update_individual_graph callback.
    """
    print("🎯 EVENT ZOOM HANDLER!")
    print(f"Graph store type: {type(graph_store_data)}")
    print(f"Graph store: {graph_store_data}")

    if graph_store_data:
        print(f"Graph store length: {len(graph_store_data)}")
        if len(graph_store_data) > 0:
            print(f"First element type: {type(graph_store_data[0])}")
            print(f"First element: {graph_store_data[0]}")

    print(f"Button IDs: {len(button_ids) if button_ids else 0}")
    print(f"Events store IDs: {len(events_store_ids) if events_store_ids else 0}")

    # Check which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        print("No trigger detected")
        return dash.no_update

    triggered_id = ctx.triggered[0]['prop_id']
    print(f"Triggered by: {triggered_id}")

    # Parse the button info
    import json
    try:
        button_info = json.loads(triggered_id.split('.')[0])
        button_panel_id = button_info['index'].split('_')[0]
        button_index = int(button_info['index'].split('_')[1])

        print(f"🎯 BUTTON CLICK: Panel {button_panel_id}, Event {button_index}")

        # Validate graph_store_data structure
        if not graph_store_data or not isinstance(graph_store_data, list):
            print("❌ Graph store data is not a valid list")
            return dash.no_update

        # Find the corresponding panel in graph store
        target_panel = None
        for panel in graph_store_data:
            if isinstance(panel, dict) and panel.get('id') == button_panel_id:
                target_panel = panel
                break

        if not target_panel:
            print(f"❌ Panel {button_panel_id} not found in graph store")
            return dash.no_update

        print(f"✅ Found panel {button_panel_id}")

        # Find the corresponding events data by matching panel IDs
        events_data = None
        for i, store_id in enumerate(events_store_ids):
            if store_id['index'] == button_panel_id:
                events_data = events_data_list[i]
                break

        if not events_data or button_index >= len(events_data):
            print(f"❌ Invalid event index {button_index} for {len(events_data) if events_data else 0} events")
            return dash.no_update

        event = events_data[button_index]
        print(f"Event: {event.get('name', 'Unknown')}")

        # Parse event times
        start_time = pd.to_datetime(event['start'])
        end_time = pd.to_datetime(event['end'])
        print(f"Event time: {start_time} to {end_time}")

        # Calculate zoom range
        duration = end_time - start_time
        if duration.total_seconds() == 0:
            padding = pd.Timedelta(hours=1)
        else:
            padding = duration * 0.5

        zoom_start = start_time - padding
        zoom_end = end_time + padding

        print(f"Zoom range: {zoom_start} - {zoom_end}")

        # Create zoom info to store in panel data
        target_panel['zoom_event'] = {
            'start': zoom_start.isoformat(),
            'end': zoom_end.isoformat(),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        print("✅ Zoom info stored in panel data")
        return graph_store_data

    except Exception as e:
        print(f"❌ Error in handle_event_zoom: {e}")
        import traceback
        traceback.print_exc()
        return dash.no_update



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
