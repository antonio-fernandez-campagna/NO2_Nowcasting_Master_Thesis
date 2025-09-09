"""
Utilidades de procesamiento de datos para modelos de predicción de NO2.

Este módulo contiene todas las funciones comunes de carga, preprocesamiento,
limpieza y transformación de datos utilizadas por los diferentes algoritmos.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import warnings

from ..config import FILE_PATHS, COLUMNS_FOR_OUTLIERS

warnings.filterwarnings('ignore')


# ==================== CARGA DE DATOS ====================

@st.cache_data(ttl=3600)
def load_master_data() -> pd.DataFrame:
    """
    Carga el dataset principal con cache para optimizar rendimiento.
    
    Returns:
        DataFrame con los datos principales o DataFrame vacío si hay error
    """
    try:
        df = pd.read_parquet(FILE_PATHS['data'])
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame()


# ==================== TRANSFORMACIONES TEMPORALES ====================

def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables cíclicas para capturar patrones temporales.
    
    Esta función es compatible tanto con XGBoost como con otros algoritmos,
    generando variables sin/cos para representar ciclos temporales de manera continua.
    
    Args:
        df: DataFrame con columna 'fecha'
        
    Returns:
        DataFrame con variables cíclicas añadidas
    """
    df = df.copy()

    # Crear variables temporales base
    df['day_of_week'] = df['fecha'].dt.dayofweek
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['hour'] = df['fecha'].dt.hour
    df['day'] = df['fecha'].dt.day
    
    # Crear variable estacional numérica (0-3: winter, spring, summer, autumn)
    df['season'] = df['month'].apply(
        lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3
    )
    
    # Variables cíclicas temporales básicas
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Variables cíclicas adicionales más específicas
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternativa simplificada para crear features temporales.
    Alias para create_cyclical_features para compatibilidad.
    
    Args:
        df: DataFrame con columna 'fecha'
        
    Returns:
        DataFrame con variables temporales añadidas
    """
    return create_cyclical_features(df)


# ==================== FILTRADO DE OUTLIERS ====================

def remove_outliers(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Elimina outliers según el método especificado.
    
    Args:
        df: DataFrame con los datos
        method: Método de detección ('iqr', 'zscore', 'quantiles', 'none')
        
    Returns:
        DataFrame filtrado sin outliers
    """
    if method == 'none':
        return df
    
    df_filtered = df.copy()
    
    if method == 'iqr':
        for col in COLUMNS_FOR_OUTLIERS:
            if col in df_filtered.columns:
                Q1 = df_filtered[col].quantile(0.25)
                Q3 = df_filtered[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]
    
    elif method == 'zscore':
        for col in COLUMNS_FOR_OUTLIERS:
            if col in df_filtered.columns:
                z_scores = zscore(df_filtered[col], nan_policy='omit')
                df_filtered = df_filtered[np.abs(z_scores) < 3.0]
    
    elif method == 'quantiles':
        for col in COLUMNS_FOR_OUTLIERS:
            if col in df_filtered.columns:
                lower = df_filtered[col].quantile(0.01)
                upper = df_filtered[col].quantile(0.99)
                df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]
    
    return df_filtered


# ==================== DIVISIÓN DE DATOS ====================

def split_data(df: pd.DataFrame, split_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba basado en fecha.
    
    Args:
        df: DataFrame con columna 'fecha'
        split_date: Fecha de división
        
    Returns:
        Tupla con (train_df, test_df)
    """
    train = df[df['fecha'] < split_date].copy()
    test = df[df['fecha'] >= split_date].copy()
    return train, test


# ==================== ESCALADO DE DATOS ====================

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Escala las variables predictoras usando StandardScaler.
    
    Args:
        X_train: Variables predictoras de entrenamiento
        X_test: Variables predictoras de prueba
        features: Lista de nombres de variables a escalar
        
    Returns:
        Tupla con (X_train_scaled, X_test_scaled, scaler_dict)
    """
    scaler_dict = {}
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    for feature in features:
        if feature in X_train.columns and pd.api.types.is_numeric_dtype(X_train[feature]):
            scaler = StandardScaler()
            X_train_scaled[feature] = scaler.fit_transform(X_train[[feature]]).flatten()
            X_test_scaled[feature] = scaler.transform(X_test[[feature]]).flatten()
            scaler_dict[feature] = scaler
    
    return X_train_scaled, X_test_scaled, scaler_dict


def scale_target(y_train: pd.Series) -> Tuple[np.ndarray, StandardScaler]:
    """
    Escala la variable objetivo usando StandardScaler.
    
    Args:
        y_train: Variable objetivo de entrenamiento
        
    Returns:
        Tupla con (y_scaled, scaler)
    """
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    return y_scaled, scaler


# ==================== PREPARACIÓN DE DATOS PARA MODELOS ESPECÍFICOS ====================

def prepare_data_for_training(df: pd.DataFrame, selected_features: List[str], 
                            split_date: pd.Timestamp, outlier_method: str = 'none',
                            preprocessing: str = 'none') -> Dict:
    """
    Pipeline completo de preparación de datos para entrenamiento.
    
    Args:
        df: DataFrame con los datos
        selected_features: Lista de variables seleccionadas
        split_date: Fecha de división
        outlier_method: Método de filtrado de outliers
        preprocessing: Tipo de preprocesamiento
        
    Returns:
        Diccionario con datos procesados y metadatos
    """
    df_processed = df.copy()
    
    # Aplicar preprocesamiento temporal
    if preprocessing == 'sin_cos':
        df_processed = create_cyclical_features(df_processed)
        
    # Dividir datos ANTES de eliminar outliers
    train_df, test_df = split_data(df_processed, split_date)
    
    # Eliminar outliers SOLO del conjunto de entrenamiento
    outliers_removed = 0
    if outlier_method != 'none':
        len_before = len(train_df)
        train_df = remove_outliers(train_df, outlier_method)
        outliers_removed = len_before - len(train_df)
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'outliers_removed': outliers_removed,
        'total_samples': len(df_processed),
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }


def prepare_matrices_for_training(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                selected_features: List[str], target_col: str = 'no2_value') -> Dict:
    """
    Prepara matrices X, y limpias para entrenamiento.
    
    Args:
        train_df: DataFrame de entrenamiento
        test_df: DataFrame de prueba
        selected_features: Lista de variables seleccionadas
        target_col: Nombre de la columna objetivo
        
    Returns:
        Diccionario con matrices preparadas
    """
    # Preparar matrices
    X_train = train_df[selected_features].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df[target_col].copy()
    
    # Limpiar NaNs
    train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
    
    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    X_test_clean = X_test[test_mask]
    y_test_clean = y_test[test_mask]
    test_df_clean = test_df[test_mask]
    
    return {
        'X_train': X_train_clean,
        'y_train': y_train_clean,
        'X_test': X_test_clean,
        'y_test': y_test_clean,
        'test_df': test_df_clean,
        'train_samples_clean': len(X_train_clean),
        'test_samples_clean': len(X_test_clean)
    }


# ==================== FUNCIONES DE VALIDACIÓN ====================

def validate_data_quality(df: pd.DataFrame, selected_features: List[str]) -> Dict:
    """
    Valida la calidad de los datos seleccionados.
    
    Args:
        df: DataFrame a validar
        selected_features: Lista de características seleccionadas
        
    Returns:
        Diccionario con métricas de calidad
    """
    quality_report = {
        'total_samples': len(df),
        'missing_features': [],
        'features_with_nulls': {},
        'numeric_features': [],
        'categorical_features': [],
        'constant_features': [],
        'high_correlation_pairs': []
    }
    
    # Verificar características faltantes
    for feature in selected_features:
        if feature not in df.columns:
            quality_report['missing_features'].append(feature)
    
    # Analizar características existentes
    existing_features = [f for f in selected_features if f in df.columns]
    
    for feature in existing_features:
        # Contar nulls
        null_count = df[feature].isnull().sum()
        if null_count > 0:
            quality_report['features_with_nulls'][feature] = {
                'count': null_count,
                'percentage': (null_count / len(df)) * 100
            }
        
        # Clasificar tipo de dato
        if pd.api.types.is_numeric_dtype(df[feature]):
            quality_report['numeric_features'].append(feature)
            
            # Verificar si es constante
            if df[feature].nunique() <= 1:
                quality_report['constant_features'].append(feature)
        else:
            quality_report['categorical_features'].append(feature)
    
    return quality_report


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Obtiene resumen estadístico de los datos.
    
    Args:
        df: DataFrame a resumir
        
    Returns:
        Diccionario con estadísticas descriptivas
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'date_range': None,
        'sensors': {
            'no2_sensors': df['id_no2'].nunique() if 'id_no2' in df.columns else 0,
            'traffic_sensors': df['id_trafico'].nunique() if 'id_trafico' in df.columns else 0
        }
    }
    
    if 'fecha' in df.columns:
        summary['date_range'] = {
            'start': df['fecha'].min(),
            'end': df['fecha'].max(),
            'days': (df['fecha'].max() - df['fecha'].min()).days
        }
    
    return summary 