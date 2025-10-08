import pandas as pd
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from typing import Dict, List, Optional


class TimeSeriesAnomalyDetector:
    """
    Clase para detectar anomalías en series de tiempo utilizando diferentes métodos.
    Inicialmente implementa Isolation Forest, pero está diseñada para ser extensible.
    """

    def __init__(self, series_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Inicializa el detector de anomalías.

        Args:
            series_data: Diccionario opcional con series de tiempo.
                        Llave: nombre de la serie, Valor: DataFrame de Pandas.
                        El índice debe ser de tipo datetime.
        """
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, pd.DataFrame] = {}

        if series_data:
            for name, df in series_data.items():
                self.add_series(name, df)

    def add_series(self, name: str, df: pd.DataFrame) -> None:
        """
        Agrega una nueva serie de tiempo al detector.

        Args:
            name: Nombre identificador de la serie
            df: DataFrame de Pandas con la serie de tiempo.
                Debe tener un índice de tipo datetime.
        """
        # Limpiar y convertir valores numéricos si es necesario
        df = self._clean_numeric_columns(df)

        # Asegurar que el índice sea datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # Si no tiene índice datetime, intentar convertir la primera columna
            if len(df.columns) > 0:
                first_col = df.columns[0]
                try:
                    df = df.set_index(pd.to_datetime(df[first_col]))
                    df = df.drop(columns=[first_col])
                except:
                    raise ValueError(f"No se pudo convertir el índice a datetime para la serie {name}")

        self.dataframes[name] = df.copy()
        self.results[name] = pd.DataFrame(index=df.index)

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia columnas numéricas que pueden tener formatos problemáticos.

        Args:
            df: DataFrame a limpiar

        Returns:
            DataFrame con columnas numéricas limpiadas
        """
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Intentar convertir a numérico
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    # Si falla, intentar limpieza manual para valores con separadores de miles
                    def clean_value(val):
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            if isinstance(val, str):
                                parts = val.split('.')
                                if len(parts) > 2:
                                    # Mantener el último punto como decimal
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

        # Eliminar filas con NaN en todas las columnas numéricas
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_clean = df_clean.dropna(subset=numeric_cols)

        return df_clean

    def apply_isolation_forest(self, series_name: str, target_col: str,
                             n_estimators: int = 100, contamination: float = 0.01) -> None:
        """
        Aplica el método Isolation Forest para detectar anomalías.

        Args:
            series_name: Nombre de la serie a analizar
            target_col: Nombre de la columna objetivo
            n_estimators: Número de estimadores para IsolationForest
            contamination: Proporción esperada de anomalías
        """
        if series_name not in self.dataframes:
            raise ValueError(f"Serie '{series_name}' no encontrada")

        df = self.dataframes[series_name]
        if target_col not in df.columns:
            raise ValueError(f"Columna '{target_col}' no encontrada en la serie '{series_name}'")

        # Feature Engineering: Crear features de rezago
        data = df[[target_col]].copy()
        data.columns = ['value']

        # Crear lags (t-1, t-2, t-3)
        for lag in range(1, 4):
            data[f'lag_{lag}'] = data['value'].shift(lag)

        # Eliminar filas con NaN (primeras filas sin lags)
        data_clean = data.dropna()

        if len(data_clean) < 10:
            raise ValueError("No hay suficientes datos para aplicar Isolation Forest")

        # Preparar features para el modelo
        features = data_clean[['value', 'lag_1', 'lag_2', 'lag_3']]

        # Entrenar IsolationForest
        iforest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
        )

        # Predicción: -1 para anomalía, 1 para normal
        predictions = iforest.fit_predict(features)
        scores = iforest.score_samples(features)

        # Crear DataFrame con resultados alineados con el índice original
        result_df = pd.DataFrame({
            'is_anomaly_IF': np.nan
        }, index=data.index)

        # Asignar predicciones solo para las filas que se usaron en el modelo
        result_df.loc[data_clean.index, 'is_anomaly_IF'] = predictions
        result_df.loc[data_clean.index, 'anomaly_score_IF'] = scores

        # Si no existe results para esta serie, inicializarlo
        if series_name not in self.results:
            self.results[series_name] = result_df
        else:
            # Combinar con resultados existentes
            self.results[series_name] = pd.concat([
                self.results[series_name], result_df
            ], axis=1)

    def apply_method(self, series_name: str, method_name: str, **kwargs) -> None:
        """
        Método placeholder para futuras implementaciones de otros algoritmos
        de detección de anomalías.

        Args:
            series_name: Nombre de la serie a analizar
            method_name: Nombre del método a aplicar
            **kwargs: Parámetros específicos del método
        """
        # Este es un placeholder para futuras extensiones
        # Por ejemplo: apply_one_class_svm, apply_hampel_filter, etc.
        raise NotImplementedError(f"Método '{method_name}' aún no implementado")

    def plot_anomalies(self, series_name: str, target_col: str,
                      methods_to_plot: List[str]) -> go.Figure:
        """
        Crea una visualización interactiva de la serie de tiempo con anomalías detectadas.

        Args:
            series_name: Nombre de la serie a visualizar
            target_col: Nombre de la columna objetivo
            methods_to_plot: Lista de métodos cuyas anomalías mostrar

        Returns:
            Figura de Plotly con la visualización
        """
        if series_name not in self.dataframes:
            raise ValueError(f"Serie '{series_name}' no encontrada")

        df = self.dataframes[series_name]
        results = self.results.get(series_name, pd.DataFrame())

        # Crear figura base
        fig = go.Figure()

        # Agregar la serie de tiempo principal
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[target_col],
            mode='lines',
            name=f'{series_name} - {target_col}',
            line=dict(color='blue', width=2)
        ))

        # Colores para diferentes métodos
        colors = ['red', 'orange', 'green', 'purple', 'brown']

        # Agregar anomalías detectadas por cada método
        for i, method in enumerate(methods_to_plot):
            anomaly_col = f'is_anomaly_{method}'

            if anomaly_col not in results.columns:
                print(f"Advertencia: Columna '{anomaly_col}' no encontrada en resultados")
                continue

            # Filtrar solo anomalías (-1)
            anomalies = results[results[anomaly_col] == -1]

            if len(anomalies) > 0:
                # Agregar puntos de anomalías
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=df.loc[anomalies.index, target_col],
                    mode='markers',
                    name=f'Anomalías - {method}',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,
                        symbol='x'
                    )
                ))

        # Configurar layout
        fig.update_layout(
            title=f'Análisis de Anomalías - {series_name}',
            xaxis_title='Tiempo',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def plot_multiple_series(self, series_names: List[str], target_col: str,
                           methods_to_plot: List[str]) -> go.Figure:
        """
        Crea una visualización interactiva combinada de múltiples series de tiempo con anomalías detectadas.

        Args:
            series_names: Lista de nombres de las series a visualizar
            target_col: Nombre de la columna objetivo
            methods_to_plot: Lista de métodos cuyas anomalías mostrar

        Returns:
            Figura de Plotly con la visualización combinada
        """
        # Crear figura base
        fig = go.Figure()

        # Colores para diferentes series
        series_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        anomaly_colors = ['red', 'darkgreen', 'darkorange', 'darkviolet', 'darkred', 'hotpink', 'dimgray', 'darkolivegreen']

        for i, series_name in enumerate(series_names):
            if series_name not in self.dataframes:
                print(f"Advertencia: Serie '{series_name}' no encontrada")
                continue

            df = self.dataframes[series_name]
            results = self.results.get(series_name, pd.DataFrame())

            # Color para esta serie
            color_idx = i % len(series_colors)
            series_color = series_colors[color_idx]
            anomaly_color = anomaly_colors[color_idx]

            # Agregar la serie de tiempo principal
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[target_col],
                mode='lines',
                name=f'{series_name} - {target_col}',
                line=dict(color=series_color, width=2),
                legendgroup=series_name
            ))

            # Agregar anomalías detectadas por cada método
            for method in methods_to_plot:
                anomaly_col = f'is_anomaly_{method}'

                if anomaly_col not in results.columns:
                    print(f"Advertencia: Columna '{anomaly_col}' no encontrada en resultados de {series_name}")
                    continue

                # Filtrar solo anomalías (-1)
                anomalies = results[results[anomaly_col] == -1]

                if len(anomalies) > 0:
                    # Agregar puntos de anomalías
                    fig.add_trace(go.Scatter(
                        x=anomalies.index,
                        y=df.loc[anomalies.index, target_col],
                        mode='markers',
                        name=f'Anomalías {series_name} - {method}',
                        marker=dict(
                            color=anomaly_color,
                            size=8,
                            symbol='x'
                        ),
                        legendgroup=series_name,
                        showlegend=True
                    ))

        # Configurar layout
        series_list = ", ".join(series_names)
        fig.update_layout(
            title=f'Análisis de Anomalías - Series: {series_list}',
            xaxis_title='Tiempo',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True,
            height=600
        )

        return fig

    def load_series_from_csv(self, file_path: str, target_col: str = 'Value') -> List[Dict[str, str]]:
        """
        Carga series de tiempo desde un CSV grande usando Polars para eficiencia.
        Agrupa los datos por WebId, Id, Name y los añade al detector.

        Args:
            file_path: Ruta al archivo CSV.
            target_col: Nombre de la columna de valor (por defecto 'Value').

        Returns:
            Una lista de diccionarios con opciones para el selector de series de Dash.
            Formato: [{'label': 'Id_Name', 'value': 'WebId'}, ...]
        """
        # 1. Carga Eficiente y Filtro (Usar Polars)
        # Cargar solo las columnas necesarias
        columns_to_load = ['WebId', 'Id', 'Name', 'Timestamp', target_col]

        try:
            df_pl = pl.read_csv(file_path, columns=columns_to_load)
        except Exception as e:
            raise ValueError(f"Error al cargar el archivo CSV: {e}")

        # 2. Conversión de Tipos (Polars)
        # Asegurar el tipo de dato para Timestamp
        df_pl = df_pl.with_columns(
            pl.col("Timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S").alias("Timestamp")
        )

        # 3. Agrupación y Generación de Series (Polars)
        # Agrupar por WebId, Id, Name
        options_list = []

        # Obtener grupos únicos
        grouped = df_pl.group_by(['WebId', 'Id', 'Name'])

        for group_keys, group_df_pl in grouped:
            web_id, id_, name_ = group_keys

            # 4. Generación de Metadata y Pandas DataFrame
            series_name = f"{id_}_{name_}"

            # Convertir a Pandas DataFrame para compatibilidad con el resto de la clase
            df_pd = group_df_pl.select(['Timestamp', target_col]).to_pandas()
            df_pd.set_index('Timestamp', inplace=True)

            # 5. Agregar la Serie y la Opción de Dash
            self.add_series(str(web_id), df_pd)  # Usar WebId como clave

            options_list.append({
                'label': series_name,  # ID_Name para el texto del selector (label de Dash)
                'value': str(web_id)   # WebId para la clave interna (value de Dash)
            })

        return options_list
