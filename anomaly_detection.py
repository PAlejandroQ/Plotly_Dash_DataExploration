import pandas as pd
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from typing import Dict, List, Optional


class TimeSeriesAnomalyDetector:
    """
    Class to detect anomalies in time series using different methods.
    Initially implements Isolation Forest, but is designed to be extensible.
    """

    def __init__(self, series_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initializes the anomaly detector.

        Args:
            series_data: Optional dictionary with time series.
                        Key: series name, Value: Pandas DataFrame.
                        The index must be of type datetime.
        """
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, pd.DataFrame] = {}

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
        self.results[name] = pd.DataFrame(index=df.index)

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

    def apply_isolation_forest(self, series_name: str, target_col: str,
                             n_estimators: int = 100, contamination: float = 0.01) -> None:
        """
        Applies the Isolation Forest method to detect anomalies.

        Args:
            series_name: Name of the series to analyze
            target_col: Name of the target column
            n_estimators: Number of estimators for IsolationForest
            contamination: Proportion of expected anomalies
        """
        if series_name not in self.dataframes:
            raise ValueError(f"Series '{series_name}' not found")

        df = self.dataframes[series_name]
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in series '{series_name}'")

        # Feature Engineering: Create lag features
        data = df[[target_col]].copy()
        data.columns = ['value']

        # Create lags (t-1, t-2, t-3)
        for lag in range(1, 4):
            data[f'lag_{lag}'] = data['value'].shift(lag)

        # Drop rows with NaN (first rows without lags)
        data_clean = data.dropna()

        if len(data_clean) < 10:
            raise ValueError("Not enough data to apply Isolation Forest")

        # Prepare features for the model
        features = data_clean[['value', 'lag_1', 'lag_2', 'lag_3']]

        # Train IsolationForest
        iforest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
        )

        # Prediction: -1 for anomaly, 1 for normal
        predictions = iforest.fit_predict(features)
        scores = iforest.score_samples(features)

        # Create DataFrame with results aligned with the original index
        result_df = pd.DataFrame({
            'is_anomaly_IF': np.nan
        }, index=data.index)

        # Assign predictions only for rows used in the model
        result_df.loc[data_clean.index, 'is_anomaly_IF'] = predictions
        result_df.loc[data_clean.index, 'anomaly_score_IF'] = scores

        # If results for this series do not exist, initialize it
        if series_name not in self.results:
            self.results[series_name] = result_df
        else:
            # Combine with existing results
            self.results[series_name] = pd.concat([
                self.results[series_name], result_df
            ], axis=1)

    def apply_method(self, series_name: str, method_name: str, **kwargs) -> None:
        """
        Placeholder method for future implementations of other anomaly detection algorithms.

        Args:
            series_name: Name of the series to analyze
            method_name: Name of the method to apply
            **kwargs: Specific parameters for the method
        """
        # This is a placeholder for future extensions
        # For example: apply_one_class_svm, apply_hampel_filter, etc.
        raise NotImplementedError(f"Method '{method_name}' not yet implemented")

    def plot_anomalies(self, series_name: str, target_col: str,
                      methods_to_plot: List[str]) -> go.Figure:
        """
        Creates an interactive visualization of the time series with detected anomalies.

        Args:
            series_name: Name of the series to visualize
            target_col: Name of the target column
            methods_to_plot: List of methods whose anomalies to show

        Returns:
            Plotly figure with the visualization
        """
        if series_name not in self.dataframes:
            raise ValueError(f"Series '{series_name}' not found")

        df = self.dataframes[series_name]
        results = self.results.get(series_name, pd.DataFrame())

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

        # Colors for different methods
        colors = ['red', 'orange', 'green', 'purple', 'brown']

        # Add detected anomalies by each method
        for i, method in enumerate(methods_to_plot):
            anomaly_col = f'is_anomaly_{method}'

            if anomaly_col not in results.columns:
                print(f"Warning: Column '{anomaly_col}' not found in results")
                continue

            # Filter only anomalies (-1)
            anomalies = results[results[anomaly_col] == -1]

            if len(anomalies) > 0:
                # Add anomaly points
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=df.loc[anomalies.index, target_col],
                    mode='markers',
                    name=f'Anomalies - {method}',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,
                        symbol='x'
                    )
                ))

        # Configure layout
        fig.update_layout(
            title=f'Anomaly Analysis - {series_name}',
            xaxis_title='Time',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def plot_multiple_series(self, series_names: List[str], target_col: str,
                           methods_to_plot: List[str]) -> go.Figure:
        """
        Creates an interactive combined visualization of multiple time series with detected anomalies.

        Args:
            series_names: List of names of the series to visualize
            target_col: Name of the target column
            methods_to_plot: List of methods whose anomalies to show

        Returns:
            Plotly figure with the combined visualization
        """
        # Create base figure
        fig = go.Figure()

        # Colors for different series
        series_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        anomaly_colors = ['red', 'darkgreen', 'darkorange', 'darkviolet', 'darkred', 'hotpink', 'dimgray', 'darkolivegreen']

        for i, series_name in enumerate(series_names):
            if series_name not in self.dataframes:
                print(f"Warning: Series '{series_name}' not found")
                continue

            df = self.dataframes[series_name]
            results = self.results.get(series_name, pd.DataFrame())

            # Color for this series
            color_idx = i % len(series_colors)
            series_color = series_colors[color_idx]
            anomaly_color = anomaly_colors[color_idx]

            # Add the main time series
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[target_col],
                mode='lines',
                name=f'{series_name} - {target_col}',
                line=dict(color=series_color, width=2),
                legendgroup=series_name
            ))

            # Add detected anomalies by each method
            for method in methods_to_plot:
                anomaly_col = f'is_anomaly_{method}'

                if anomaly_col not in results.columns:
                    print(f"Warning: Column '{anomaly_col}' not found in results for {series_name}")
                    continue

                # Filter only anomalies (-1)
                anomalies = results[results[anomaly_col] == -1]

                if len(anomalies) > 0:
                    # Add anomaly points
                    fig.add_trace(go.Scatter(
                        x=anomalies.index,
                        y=df.loc[anomalies.index, target_col],
                        mode='markers',
                        name=f'Anomalies {series_name} - {method}',
                        marker=dict(
                            color=anomaly_color,
                            size=8,
                            symbol='x'
                        ),
                        legendgroup=series_name,
                        showlegend=True
                    ))

        # Configure layout
        series_list = ", ".join(series_names)
        fig.update_layout(
            title=f'Anomaly Analysis - Series: {series_list}',
            xaxis_title='Time',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True,
            height=600
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
        'WebId': pl.Utf8, # Asumimos Utf8 para IDs por seguridad
        'Id': pl.Utf8,
        'Name': pl.Utf8,
        'Timestamp': pl.Utf8,
        target_col: pl.Utf8
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
