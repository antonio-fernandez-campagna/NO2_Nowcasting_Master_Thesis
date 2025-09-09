"""
Utilidades comunes para la aplicación de análisis de NO₂ en Madrid.

Este módulo contiene funciones reutilizables para carga de datos,
visualización, procesamiento y otras utilidades compartidas.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os
import joblib
from datetime import datetime, timedelta
import time

from .config import DATA_PATHS, CACHE_CONFIG, STYLE_CONFIG, METRICS_CONFIG

# ==================== FUNCIONES DE CARGA DE DATOS ====================

@st.cache_data(ttl=CACHE_CONFIG['ttl'], show_spinner=CACHE_CONFIG['show_spinner'])
def load_no2_data() -> pd.DataFrame:
    """
    Carga los datos principales de NO₂.
    
    Returns:
        pd.DataFrame: DataFrame con los datos de NO₂
    """
    try:
        data_path = DATA_PATHS['NO2_DATA']
        if not os.path.exists(data_path):
            st.error(f"No se encontró el archivo de datos: {data_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(data_path)
        
        # Asegurar que la columna fecha esté en formato datetime
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
        
        return df
    
    except Exception as e:
        st.error(f"Error cargando datos de NO₂: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_CONFIG['ttl'], show_spinner=CACHE_CONFIG['show_spinner'])
def load_traffic_no2_data() -> pd.DataFrame:
    """
    Carga los datos de NO₂ con información de tráfico y meteorología.
    
    Returns:
        pd.DataFrame: DataFrame con los datos integrados
    """
    try:
        data_path = DATA_PATHS['TRAFFIC_NO2_DATA']
        if not os.path.exists(data_path):
            st.error(f"No se encontró el archivo de datos: {data_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(data_path)
        
        # Asegurar que la columna fecha esté en formato datetime
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
        
        return df
    
    except Exception as e:
        st.error(f"Error cargando datos integrados: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_CONFIG['ttl'], show_spinner=CACHE_CONFIG['show_spinner'])
def load_sensor_mapping() -> pd.DataFrame:
    """
    Carga los datos de mapeo de sensores.
    
    Returns:
        pd.DataFrame: DataFrame con el mapeo de sensores
    """
    try:
        data_path = DATA_PATHS['SENSOR_MAPPING']
        if not os.path.exists(data_path):
            st.error(f"No se encontró el archivo de mapeo: {data_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(data_path)
        return df
    
    except Exception as e:
        st.error(f"Error cargando mapeo de sensores: {str(e)}")
        return pd.DataFrame()


# ==================== FUNCIONES DE PROCESAMIENTO ====================

def filter_data_by_date(df: pd.DataFrame, start_date: datetime, end_date: datetime, 
                       date_col: str = 'fecha') -> pd.DataFrame:
    """
    Filtra un DataFrame por rango de fechas.
    
    Args:
        df: DataFrame a filtrar
        start_date: Fecha de inicio
        end_date: Fecha de fin
        date_col: Nombre de la columna de fecha
    
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    if date_col not in df.columns:
        st.warning(f"Columna '{date_col}' no encontrada en los datos")
        return df
    
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]


def filter_data_by_station(df: pd.DataFrame, stations: List[str], 
                          station_col: str = 'station_name') -> pd.DataFrame:
    """
    Filtra un DataFrame por estaciones específicas.
    
    Args:
        df: DataFrame a filtrar
        stations: Lista de nombres de estaciones
        station_col: Nombre de la columna de estación
    
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    if station_col not in df.columns:
        st.warning(f"Columna '{station_col}' no encontrada en los datos")
        return df
    
    return df[df[station_col].isin(stations)]


def calculate_statistics(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    """
    Calcula estadísticas básicas para una columna.
    
    Args:
        df: DataFrame
        value_col: Nombre de la columna de valores
    
    Returns:
        Dict con estadísticas
    """
    if value_col not in df.columns:
        return {}
    
    series = df[value_col].dropna()
    
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'count': len(series),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75)
    }


def detect_outliers(df: pd.DataFrame, value_col: str, method: str = 'iqr') -> pd.Series:
    """
    Detecta outliers en una serie usando diferentes métodos.
    
    Args:
        df: DataFrame
        value_col: Nombre de la columna de valores
        method: Método de detección ('iqr', 'zscore')
    
    Returns:
        pd.Series: Serie booleana indicando outliers
    """
    if value_col not in df.columns:
        return pd.Series(dtype=bool)
    
    series = df[value_col].dropna()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        from scipy.stats import zscore
        z_scores = np.abs(zscore(series))
        return z_scores > 3
    
    return pd.Series(dtype=bool)


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def create_time_series_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                           title: str = "", height: int = 400) -> go.Figure:
    """
    Crea un gráfico de serie temporal.
    
    Args:
        df: DataFrame con los datos
        x_col: Columna del eje X (temporal)
        y_col: Columna del eje Y (valores)
        title: Título del gráfico
        height: Altura del gráfico
    
    Returns:
        go.Figure: Figura de Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines',
        name=y_col,
        line=dict(color=STYLE_CONFIG['primary_color'])
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=height,
        showlegend=False
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: List[str], 
                              title: str = "Matriz de Correlación") -> go.Figure:
    """
    Crea un mapa de calor de correlaciones.
    
    Args:
        df: DataFrame con los datos
        columns: Lista de columnas para incluir
        title: Título del gráfico
    
    Returns:
        go.Figure: Figura de Plotly
    """
    # Filtrar columnas que existen en el DataFrame
    available_columns = [col for col in columns if col in df.columns]
    
    if not available_columns:
        st.warning("No se encontraron columnas válidas para el análisis de correlación")
        return go.Figure()
    
    # Calcular correlación
    corr_matrix = df[available_columns].corr()
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=600
    )
    
    return fig


def create_distribution_plot(df: pd.DataFrame, value_col: str, 
                           title: str = "", bins: int = 50) -> go.Figure:
    """
    Crea un histograma de distribución.
    
    Args:
        df: DataFrame con los datos
        value_col: Columna de valores
        title: Título del gráfico
        bins: Número de bins
    
    Returns:
        go.Figure: Figura de Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[value_col].dropna(),
        nbinsx=bins,
        name=value_col,
        marker_color=STYLE_CONFIG['primary_color'],
        opacity=0.7
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=value_col,
        yaxis_title="Frecuencia",
        showlegend=False
    )
    
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       color_col: str = None, title: str = "") -> go.Figure:
    """
    Crea un gráfico de dispersión.
    
    Args:
        df: DataFrame con los datos
        x_col: Columna del eje X
        y_col: Columna del eje Y
        color_col: Columna opcional para colores
        title: Título del gráfico
    
    Returns:
        go.Figure: Figura de Plotly
    """
    if color_col and color_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=title)
    
    fig.update_traces(marker=dict(opacity=0.6))
    
    return fig


# ==================== FUNCIONES DE MÉTRICAS ====================

def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación de modelos.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        Dict con métricas
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def check_no2_limits(value: float) -> Dict[str, bool]:
    """
    Verifica si un valor de NO₂ excede los límites establecidos.
    
    Args:
        value: Valor de NO₂ en µg/m³
    
    Returns:
        Dict indicando si excede cada límite
    """
    limits = METRICS_CONFIG['no2_limits']
    
    return {
        'exceeds_who_annual': value > limits['who_annual'],
        'exceeds_who_daily': value > limits['who_daily'],
        'exceeds_eu_annual': value > limits['eu_annual'],
        'exceeds_eu_hourly': value > limits['eu_hourly']
    }


# ==================== FUNCIONES DE UTILIDADES ====================

def format_number(value: float, decimals: int = 2) -> str:
    """
    Formatea un número con el número especificado de decimales.
    
    Args:
        value: Valor a formatear
        decimals: Número de decimales
    
    Returns:
        str: Número formateado
    """
    return f"{value:.{decimals}f}"


def create_download_button(df: pd.DataFrame, filename: str, button_text: str = "Descargar datos"):
    """
    Crea un botón de descarga para un DataFrame.
    
    Args:
        df: DataFrame a descargar
        filename: Nombre del archivo
        button_text: Texto del botón
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=button_text,
        data=csv,
        file_name=filename,
        mime='text/csv'
    )


def show_dataframe_info(df: pd.DataFrame, title: str = "Información del Dataset"):
    """
    Muestra información básica sobre un DataFrame.
    
    Args:
        df: DataFrame
        title: Título de la sección
    """
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filas", len(df))
    
    with col2:
        st.metric("Columnas", len(df.columns))
    
    with col3:
        st.metric("Memoria (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}")
    
    with col4:
        st.metric("Valores Nulos", df.isnull().sum().sum())


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    División segura que maneja división por cero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si denominador es 0
    
    Returns:
        float: Resultado de la división o valor por defecto
    """
    return numerator / denominator if denominator != 0 else default


def get_station_list(df: pd.DataFrame, station_col: str = 'station_name') -> List[str]:
    """
    Obtiene lista única de estaciones de un DataFrame.
    
    Args:
        df: DataFrame
        station_col: Nombre de la columna de estación
    
    Returns:
        List[str]: Lista de nombres de estaciones
    """
    if station_col not in df.columns:
        return []
    
    return sorted(df[station_col].dropna().unique().tolist())


# ==================== FUNCIONES DE MANEJO DE ERRORES ====================

def handle_data_loading_error(error: Exception, data_type: str):
    """
    Maneja errores de carga de datos de forma consistente.
    
    Args:
        error: Excepción capturada
        data_type: Tipo de datos que se intentaba cargar
    """
    error_message = f"Error cargando {data_type}: {str(error)}"
    st.error(error_message)
    
    with st.expander("Detalles del error"):
        st.code(str(error))
        st.info("Verifica que los archivos de datos estén en la ubicación correcta y tengan el formato esperado.")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                      data_name: str = "datos") -> bool:
    """
    Valida que un DataFrame tenga las columnas requeridas.
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de columnas requeridas
        data_name: Nombre descriptivo de los datos
    
    Returns:
        bool: True si es válido, False en caso contrario
    """
    if df.empty:
        st.error(f"El DataFrame de {data_name} está vacío")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Columnas faltantes en {data_name}: {missing_columns}")
        return False
    
    return True


# ==================== FUNCIONES DE FEATURE ENGINEERING ====================

def create_lag_features(df: pd.DataFrame, 
                       variables: List[str] = None,
                       lags: List[int] = None,
                       groupby_col: str = 'id_no2',
                       sort_col: str = 'fecha') -> pd.DataFrame:
    """
    Crea variables lag para variables específicas agrupadas por sensor.
    
    NOTA: Esta función es para preprocessing de datos. Si los datos ya contienen
    variables lag (como en los archivos procesados), no es necesario usarla.
    
    Args:
        df: DataFrame con los datos
        variables: Lista de variables para crear lags. Por defecto ['intensidad', 'ocupacion', 'carga']
        lags: Lista de valores de lag. Por defecto [1, 2, 3, 4, 6, 8]
        groupby_col: Columna para agrupar (por defecto 'id_no2')
        sort_col: Columna para ordenar (por defecto 'fecha')
    
    Returns:
        pd.DataFrame: DataFrame con las variables lag añadidas
    """
    if variables is None:
        variables = ['intensidad', 'ocupacion', 'carga']
    
    if lags is None:
        lags = [1, 2, 3, 4, 6, 8]
    
    df = df.copy()
    
    # Ordenar por sensor y por fecha
    df = df.sort_values([groupby_col, sort_col])
    
    # Generar lags por sensor
    for var in variables:
        if var in df.columns:
            for lag in lags:
                df[f'{var}_lag{lag}'] = df.groupby(groupby_col)[var].shift(lag)
        else:
            st.warning(f"Variable '{var}' no encontrada en el DataFrame")
    
    return df


def create_temporal_features(df: pd.DataFrame, date_col: str = 'fecha') -> pd.DataFrame:
    """
    Crea variables temporales básicas a partir de una columna de fecha.
    
    Args:
        df: DataFrame con los datos
        date_col: Nombre de la columna de fecha
    
    Returns:
        pd.DataFrame: DataFrame con las variables temporales añadidas
    """
    df = df.copy()
    
    # Asegurar que la columna de fecha esté en formato datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Crear variables temporales
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Crear variable estacional (0=invierno, 1=primavera, 2=verano, 3=otoño)
    df['season'] = df['month'].apply(
        lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3
    )
    
    df['hour'] = df[date_col].dt.hour
    df['day'] = df[date_col].dt.day
    
    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables cíclicas para capturar patrones temporales.
    
    Args:
        df: DataFrame que debe contener variables temporales básicas
    
    Returns:
        pd.DataFrame: DataFrame con variables cíclicas añadidas
    """
    df = df.copy()
    
    # Variables cíclicas básicas
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'day' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    if 'day_of_week' in df.columns:
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    if 'day_of_year' in df.columns:
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    if 'season' in df.columns:
        df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
        df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    
    return df


def apply_full_feature_engineering(df: pd.DataFrame,
                                 create_lags: bool = True,
                                 lag_variables: List[str] = None,
                                 lag_values: List[int] = None,
                                 create_temporal: bool = True,
                                 create_cyclical: bool = True,
                                 groupby_col: str = 'id_no2',
                                 date_col: str = 'fecha') -> pd.DataFrame:
    """
    Aplica el pipeline completo de feature engineering.
    
    Args:
        df: DataFrame original
        create_lags: Si crear variables lag
        lag_variables: Variables para crear lags
        lag_values: Valores de lag
        create_temporal: Si crear variables temporales básicas
        create_cyclical: Si crear variables cíclicas
        groupby_col: Columna para agrupar en lags
        date_col: Columna de fecha
    
    Returns:
        pd.DataFrame: DataFrame con todas las características creadas
    """
    df_processed = df.copy()
    
    if create_temporal:
        df_processed = create_temporal_features(df_processed, date_col)
    
    if create_cyclical:
        df_processed = create_cyclical_features(df_processed)
    
    if create_lags:
        df_processed = create_lag_features(
            df_processed, 
            variables=lag_variables,
            lags=lag_values,
            groupby_col=groupby_col,
            sort_col=date_col
        )
    
    return df_processed 