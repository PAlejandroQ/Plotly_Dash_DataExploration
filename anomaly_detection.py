import pandas as pd
import numpy as np
import polars as pl
import plotly.graph_objects as go
from typing import Dict, List, Optional


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
                           start_date=None, end_date=None) -> go.Figure:
        """
        Creates an interactive combined visualization of multiple time series.

        Args:
            series_names: List of names of the series to visualize
            target_col: Name of the target column
            start_date: Optional start date to filter the data
            end_date: Optional end date to filter the data

        Returns:
            Plotly figure with the combined visualization
        """
        # Create base figure
        fig = go.Figure()

        # Colors for different series
        series_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']

        for i, series_name in enumerate(series_names):
            if series_name not in self.dataframes:
                print(f"Warning: Series '{series_name}' not found")
                continue

            df = self.dataframes[series_name]

            # Filter by date range if provided
            if start_date is not None and end_date is not None:
                df = df[(df.index >= start_date) & (df.index <= end_date)]

            # Color for this series
            color_idx = i % len(series_colors)
            series_color = series_colors[color_idx]

            # Add the main time series
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[target_col],
                mode='lines',
                name=f'{series_name} - {target_col}',
                line=dict(color=series_color, width=2),
                legendgroup=series_name
            ))

        # Configure layout
        series_list = ", ".join(series_names)
        fig.update_layout(
            title=f'Time Series - Series: {series_list}',
            xaxis_title='Time',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True,
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.9)', 
                bordercolor="Black",
                borderwidth=1
            )
        )

        return fig

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
