import pandas as pd
import numpy as np
import polars as pl
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime


class TimeSeriesAnomalyDetector:
    """
    Class to load and visualize time series data from Parquet files.
    """

    def __init__(self, series_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initializes the time series detector.

        Args:
            series_data: Optional dictionary with time series.
                        Key: series name, Value: Pandas DataFrame.
                        The index must be of type datetime.
        """
        self.dataframes: Dict[str, pd.DataFrame] = {}

        if series_data:
            for name, df in series_data.items():
                self.add_series(name, df)

    def _format_event_name(self, x0_str: str, x1_str: str) -> str:
        """
        Formats event name for legend display based on date range.

        Args:
            x0_str: Start time string in format '2023-02-27T02:02:00'
            x1_str: End time string in format '2023-02-27T02:02:00'

        Returns:
            Formatted string like '2023/02/27_02-2023/02/27_03' or '2023/02/27_02:15-03:15'
        """
        try:
            # Parse datetime strings
            dt0 = datetime.fromisoformat(x0_str.replace('Z', '+00:00'))
            dt1 = datetime.fromisoformat(x1_str.replace('Z', '+00:00'))

            # Format date part
            date0 = dt0.strftime('%Y/%m/%d')
            date1 = dt1.strftime('%Y/%m/%d')

            # Format time part (hours only for different dates, hours:minutes for same date)
            if date0 == date1:
                # Same date: show hours and minutes
                time0 = dt0.strftime('%H:%M')
                time1 = dt1.strftime('%H:%M')
                return f'{date0}_{time0}-{time1}'
            else:
                # Different dates: show date and hour for each
                hour0 = dt0.strftime('%H')
                hour1 = dt1.strftime('%H')
                return f'{date0}_{hour0}-{date1}_{hour1}'
        except Exception as e:
            # Fallback to original format if parsing fails
            return f'{x0_str}-{x1_str}'

    def add_series(self, name: str, df: pd.DataFrame) -> None:
        """
        Adds a new time series to the detector.

        Args:
            name: Identifier name for the series
            df: Pandas DataFrame with the time series.
                Must have a datetime index.
        """
        # Clean and convert numeric values if necessary
        df = self._clean_numeric_columns(df)

        # Ensure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # If no datetime index, try to convert the first column
            if len(df.columns) > 0:
                first_col = df.columns[0]
                try:
                    df = df.set_index(pd.to_datetime(df[first_col]))
                    df = df.drop(columns=[first_col])
                except:
                    raise ValueError(f"Could not convert index to datetime for series {name}")

        self.dataframes[name] = df.copy()

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans numeric columns that may have problematic formats.

        Args:
            df: DataFrame to clean

        Returns:
            DataFrame with cleaned numeric columns
        """
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    # If fails, try manual cleaning for values with thousand separators
                    def clean_value(val):
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            if isinstance(val, str):
                                parts = val.split('.')
                                if len(parts) > 2:
                                    # Keep the last dot as decimal
                                    decimal_part = parts[-1]
                                    integer_part = ''.join(parts[:-1])
                                    cleaned = f"{integer_part}.{decimal_part}"
                                    try:
                                        return float(cleaned)
                                    except ValueError:
                                        return np.nan
                                else:
                                    try:
                                        return float(val)
                                    except ValueError:
                                        return np.nan
                            return np.nan

                    df_clean[col] = df_clean[col].apply(clean_value)

        # Drop rows with NaN in all numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_clean = df_clean.dropna(subset=numeric_cols)

        return df_clean


    def plot_series(self, series_name: str, target_col: str) -> go.Figure:
        """
        Creates an interactive visualization of the time series.

        Args:
            series_name: Name of the series to visualize
            target_col: Name of the target column

        Returns:
            Plotly figure with the visualization
        """
        if series_name not in self.dataframes:
            raise ValueError(f"Series '{series_name}' not found")

        df = self.dataframes[series_name]

        # Create base figure
        fig = go.Figure()

        # Add the main time series
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[target_col],
            mode='lines',
            name=f'{series_name} - {target_col}',
            line=dict(color='blue', width=2)
        ))

        # Configure layout
        fig.update_layout(
            title=f'Time Series - {series_name}',
            xaxis_title='Time',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=False
        )

        return fig

    def plot_multiple_series(self, series_names: List[str], target_col: str,
                           start_date=None, end_date=None, units_dict=None, anomaly_events=None, vigres_events=None) -> tuple[go.Figure, List[Dict]]:
        """
        Creates an interactive combined visualization of multiple time series with multiple Y-axes.

        Args:
            series_names: List of names of the series to visualize
            target_col: Name of the target column
            start_date: Optional start date to filter the data
            end_date: Optional end date to filter the data
            units_dict: Dictionary mapping series names to their units
            anomaly_events: Optional list of [start, end] timestamp pairs for anomaly highlighting
            vigres_events: Optional list of [start, end] timestamp pairs for vigres event highlighting

        Returns:
            Tuple containing:
            - Plotly figure with the combined visualization
            - List of visible events with their metadata for button creation
        """
        # Create base figure
        fig = go.Figure()

        # Define Y-axis mapping for units
        unit_axis_map = {
            'kgf/cm² a': 'y',      # Primary Y-axis
            'ºC': 'y2',            # Secondary Y-axis (temperature)
            'l': 'y3',             # Third Y-axis (liters)
            'm3/d': 'y4'           # Fourth Y-axis (cubic meters per day)
        }

        # Colors for different series
        series_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        # Track which axes are used
        used_axes = set()

        # Collect visible events for button creation
        visible_events = []

        for i, series_name in enumerate(series_names):
            if series_name not in self.dataframes:
                print(f"Warning: Series '{series_name}' not found")
                continue

            df = self.dataframes[series_name]

            # Filter by date range if provided
            if start_date is not None and end_date is not None:
                df = df[(df.index >= start_date) & (df.index <= end_date)]

            # Determine which Y-axis to use
            unit = units_dict.get(series_name, '') if units_dict else ''
            yaxis = unit_axis_map.get(unit, 'y')  # Default to primary axis
            used_axes.add(yaxis)

            # Color for this series
            color_idx = i % len(series_colors)
            series_color = series_colors[color_idx]

            # Add the time series
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[target_col],
                mode='lines',
                name=f'{series_name}',
                line=dict(color=series_color, width=2),
                yaxis=yaxis,
                legendgroup=series_name
            ))

        # Configure axes based on used axes
        layout_updates = {
            'title': f'Time Series - {len(series_names)} series',
            'xaxis_title': 'Time',
            'hovermode': 'x unified',
            'showlegend': True,
            'height': 600,
            'legend': dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor="Black",
                borderwidth=1
            )
        }

        # Primary Y-axis (pressure) - left side, default position
        if 'y' in used_axes or True:
            layout_updates['yaxis'] = dict(
                title=dict(text="Pressure (kgf/cm² a)", font=dict(color="#1f77b4")),
                side="left"
            )

        # Fourth Y-axis (flow rate) - left side, to the left of pressure
        if 'y4' in used_axes:
            layout_updates['yaxis4'] = dict(
                title=dict(text="Flow Rate (m³/d)", font=dict(color="#d62728")),
                anchor="free",
                overlaying="y",
                side="left",
                position=0.07
            )

        # Secondary Y-axis (temperature) - right side
        if 'y2' in used_axes:
            layout_updates['yaxis2'] = dict(
                title=dict(text="Temperature (°C)", font=dict(color="#ff7f0e")),
                anchor="free",
                overlaying="y",
                side="right",
                position=1
            )

        # Third Y-axis (volume) - right side, extreme right
        if 'y3' in used_axes:
            layout_updates['yaxis3'] = dict(
                title=dict(text="Volume (l)", font=dict(color="#2ca02c")),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.93
            )

        fig.update_layout(**layout_updates)

        # Ensure primary Y-axis is always present by adding an invisible dummy trace if no data uses it
        if 'y' not in used_axes:
            # Find the earliest timestamp from all series being plotted to place the dummy point within range
            dummy_timestamp = None
            for series_name in series_names:
                if series_name in self.dataframes:
                    df = self.dataframes[series_name]
                    # Filter by date range if provided
                    if start_date is not None and end_date is not None:
                        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
                    else:
                        df_filtered = df

                    if len(df_filtered) > 0:
                        series_min_date = df_filtered.index.min()
                        if dummy_timestamp is None or series_min_date < dummy_timestamp:
                            dummy_timestamp = series_min_date

            # Use the earliest timestamp found, or fallback to a reasonable default
            if dummy_timestamp is None:
                dummy_timestamp = pd.Timestamp('2020-01-01')  # Fallback if no data found

            # Add an invisible dummy scatter trace to force the y-axis to appear
            fig.add_trace(go.Scatter(
                x=[dummy_timestamp],  # Use timestamp within actual data range
                y=[0],  # Dummy value
                mode='markers',
                marker=dict(size=0, color='rgba(0,0,0,0)'),  # Invisible marker
                showlegend=False,
                hoverinfo='skip',
                yaxis='y'
            ))

        # Add anomaly highlighting regions if provided
        if anomaly_events and len(anomaly_events) > 0:
            for event in anomaly_events:
                if len(event) >= 2:
                    x0_str = event[0]  # Start timestamp as string
                    x1_str = event[1]  # End timestamp as string

                    # Convert timestamps to datetime objects for comparison
                    try:
                        x0_dt = pd.to_datetime(x0_str)
                        x1_dt = pd.to_datetime(x1_str)
                    except Exception as e:
                        print(f"Warning: Could not parse anomaly event timestamps {x0_str} - {x1_str}: {e}")
                        continue

                    # Filter anomaly events by date range if provided
                    if start_date is not None and end_date is not None:
                        # Only show anomalies that overlap with the visible date range
                        if x1_dt < start_date or x0_dt > end_date:
                            print(f"Warning: Anomaly event {x0_str} - {x1_str} is outside the visible range")
                            continue  # Skip this anomaly event as it's outside the visible range
                    else:
                        continue
                    # Add vertical rectangle for anomaly region
                    fig.add_vrect(
                        x0=x0_str,  # Keep as string for Plotly
                        x1=x1_str,  # Keep as string for Plotly
                        line_width=0,
                        fillcolor="red",
                        opacity=0.2,
                        layer="below",  # Ensure it appears behind the data
                        name=self._format_event_name(x0_str, x1_str),
                        showlegend=False
                    )

                    # Collect event info for button creation
                    visible_events.append({
                        'id': f"anomaly_{len(visible_events)}",
                        'type': 'anomaly',
                        'name': self._format_event_name(x0_str, x1_str),
                        'start': x0_str,
                        'end': x1_str,
                        'color': 'red'
                    })

        # Add vigres event highlighting regions if provided
        if vigres_events and len(vigres_events) > 0:
            for event in vigres_events:
                if len(event) >= 2:
                    x0_str = event[0]  # Start timestamp as string
                    x1_str = event[1]  # End timestamp as string

                    # Convert timestamps to datetime objects for comparison
                    try:
                        x0_dt = pd.to_datetime(x0_str)
                        x1_dt = pd.to_datetime(x1_str)
                    except Exception as e:
                        print(f"Warning: Could not parse vigres event timestamps {x0_str} - {x1_str}: {e}")
                        continue

                    # Filter vigres events by date range if provided
                    if start_date is not None and end_date is not None:
                        # Only show vigres events that overlap with the visible date range
                        if x1_dt < start_date or x0_dt > end_date:
                            print(f"Warning: Vigres event {x0_str} - {x1_str} is outside the visible range")
                            continue  # Skip this vigres event as it's outside the visible range
                    else:
                        continue
                    # Add vertical rectangle for vigres event region
                    fig.add_vrect(
                        x0=x0_str,  # Keep as string for Plotly
                        x1=x1_str,  # Keep as string for Plotly
                        line_width=0,
                        fillcolor="green",
                        opacity=0.3,
                        layer="below",  # Ensure it appears behind the data
                        name=self._format_event_name(x0_str, x1_str),
                        showlegend=False
                    )

                    # Collect event info for button creation
                    visible_events.append({
                        'id': f"vigres_{len(visible_events)}",
                        'type': 'vigres',
                        'name': self._format_event_name(x0_str, x1_str),
                        'start': x0_str,
                        'end': x1_str,
                        'color': 'green'
                    })

        return fig, visible_events

    def load_series_from_csv(self, file_path: str, target_col: str = 'Value') -> List[Dict[str, str]]:
        """
        Load time series from a large CSV efficiently using Polars, clean the 'Value' and 'Timestamp' columns,
        and group by ['WebId', 'Id', 'Name'] to create series for the detector.

        Args:
            file_path: Path to the CSV file.
            target_col: Name of the value column (default 'Value').

        Returns:
            A list of dictionaries with options for the Dash selector.
            Format: [{'label': 'Id_Name', 'value': 'WebId'}, ...]
        """
        # 1. Efficient Loading and Filtering (Use Polars)
        # Load only necessary columns
        columns_to_load = ['WebId', 'Id', 'Name', 'Timestamp', target_col]

        schema = {
        'WebId': pl.Utf8,
        'Id': pl.Utf8,
        'Name': pl.Utf8,
        'Path': pl.Utf8,
        'Descriptor': pl.Utf8,
        'PointClass': pl.Utf8,
        'PointType': pl.Utf8,
        'DigitalSetName': pl.Utf8,
        'EngineeringUnits': pl.Utf8,
        'Span': pl.Utf8,
        'Zero': pl.Utf8,
        'Step': pl.Utf8,
        'Future': pl.Utf8,
        'DisplayDigits': pl.Utf8,
        'Links': pl.Utf8,
        'Timestamp': pl.Utf8, 
        'Value': pl.Utf8      
        }

        try:
            df_lazy = pl.scan_csv(file_path, 
                                            schema=schema, 
                                            infer_schema_length=0, 
                                            try_parse_dates=False,
                                            )
            df_pl = df_lazy.select(columns_to_load).collect()
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

        # 2. Convert 'Value' to float, replace non-numeric with null (strict=False)
        df_pl = df_pl.with_columns(
            pl.col(target_col).cast(pl.Float64, strict=False)
        )

        # 3. Robust timestamp parsing with ISO-8601 and variable decimal precision (and 'Z')
        #     %Y-%m-%dT%H:%M:%S.%fZ handles microseconds (or none) and trailing Z 
        df_pl = df_pl.with_columns(
            pl.col('Timestamp').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%fZ", strict=False)
        )

        # 4. Drop rows where either Value or Timestamp is null
        df_pl = df_pl.drop_nulls([target_col, 'Timestamp'])

        # 5. Group by WebId, Id, Name and build time series for each unique group
        options_list = []

        # Get unique groups
        grouped = df_pl.group_by(['WebId', 'Id', 'Name'])

        for group_keys, group_df_pl in grouped:
            web_id, id_, name_ = group_keys

            # 4. Metadata Generation and Pandas DataFrame
            series_name = f"{id_}_{name_}"
            df_pd = group_df_pl.select(['Timestamp', target_col]).to_pandas()
            df_pd.set_index('Timestamp', inplace=True)

            # 5. Add the Series and Dash Option
            self.add_series(str(web_id), df_pd)  # Use WebId as key

            options_list.append({
                'label': series_name,  # ID_Name for the selector text (label for Dash)
                'value': str(web_id)   # WebId for internal key (value for Dash)
            })

        return options_list

    def load_series_from_parquet(self, folder_path: str, target_col: str = 'Value') -> List[Dict[str, str]]:
        """
        Load time series from Parquet files in a folder. Each Parquet file contains one time series,
        and the TAG name is the filename without extension.

        Args:
            folder_path: Path to the folder containing Parquet files.
            target_col: Name of the value column (default 'Value').

        Returns:
            A list of dictionaries with options for the Dash selector.
            Format: [{'label': 'filename', 'value': 'filename'}, ...]
        """
        import os

        options_list = []
        parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

        for filename in parquet_files:
            parquet_path = os.path.join(folder_path, filename)
            series_name = os.path.splitext(filename)[0]  # Remove .parquet extension

            try:
                # Read Parquet file using Polars
                df_pl = pl.read_parquet(parquet_path)

                # Check if required columns exist
                if 'Timestamp' not in df_pl.columns or target_col not in df_pl.columns:
                    print(f"Warning: File '{filename}' missing required columns (Timestamp, {target_col}). Skipping...")
                    continue

                # Convert Timestamp to datetime if needed
                if df_pl['Timestamp'].dtype != pl.Datetime:
                    try:
                        df_pl = df_pl.with_columns(
                            pl.col('Timestamp').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
                        )
                    except:
                        try:
                            df_pl = df_pl.with_columns(
                                pl.col('Timestamp').str.strptime(pl.Datetime, strict=False)
                            )
                        except Exception as e:
                            print(f"Warning: Could not parse Timestamp in '{filename}': {e}. Skipping...")
                            continue

                # Convert Value to float
                df_pl = df_pl.with_columns(
                    pl.col(target_col).cast(pl.Float64, strict=False)
                )

                # Drop rows with null Timestamp or Value
                df_pl = df_pl.drop_nulls(['Timestamp', target_col])

                if len(df_pl) == 0:
                    print(f"Warning: File '{filename}' has no valid data after cleaning. Skipping...")
                    continue

                # Convert to pandas DataFrame with datetime index
                df_pd = df_pl.select(['Timestamp', target_col]).to_pandas()
                df_pd.set_index('Timestamp', inplace=True)

                # Add the series using the existing method
                self.add_series(series_name, df_pd)

                # Add to options list
                options_list.append({
                    'label': series_name,
                    'value': series_name
                })

                print(f"Successfully loaded series '{series_name}' from '{filename}' ({len(df_pd)} records)")

            except Exception as e:
                print(f"Error loading '{filename}': {e}. Skipping...")

        return options_list
