"""
Utilidades de gestión de session state para modelos de predicción de NO2.

Este módulo contiene todas las funciones comunes de gestión de estado de sesión,
configuración de interfaz y manejo de datos persistentes entre los diferentes algoritmos.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from ..config import (
    VARIABLE_CATEGORIES, OUTLIER_METHODS, PREPROCESSING_OPTIONS,
    get_available_features, validate_feature_selection, get_session_state_key
)


# ==================== INICIALIZACIÓN DE SESSION STATE ====================

def initialize_common_session_state(model_type: str):
    """
    Inicializa estados de sesión comunes para todos los modelos.
    
    Args:
        model_type: Tipo de modelo ('xgboost', 'gam', 'bayesian')
    """
    prefix = f"{model_type}_"
    
    # Estados básicos
    if f'{prefix}data_loaded' not in st.session_state:
        st.session_state[f'{prefix}data_loaded'] = False
    
    if f'{prefix}model_trained' not in st.session_state:
        st.session_state[f'{prefix}model_trained'] = False
    
    if f'{prefix}config' not in st.session_state:
        st.session_state[f'{prefix}config'] = {}
    
    if f'{prefix}analysis_data' not in st.session_state:
        st.session_state[f'{prefix}analysis_data'] = {}
    
    if f'{prefix}selected_features' not in st.session_state:
        st.session_state[f'{prefix}selected_features'] = []


def initialize_unified_session_state(model_type: str):
    """
    Inicializa estados de sesión para modelos unificados (individual/global).
    
    Args:
        model_type: Tipo de modelo ('xgboost', 'gam', 'bayesian')
    """
    initialize_common_session_state(model_type)
    
    prefix = f"{model_type}_unified_"
    
    if f'{prefix}mode' not in st.session_state:
        st.session_state[f'{prefix}mode'] = 'individual'
    
    if f'{prefix}individual_config' not in st.session_state:
        st.session_state[f'{prefix}individual_config'] = {}
    
    if f'{prefix}global_config' not in st.session_state:
        st.session_state[f'{prefix}global_config'] = {}
    
    if f'{prefix}individual_results' not in st.session_state:
        st.session_state[f'{prefix}individual_results'] = {}
    
    if f'{prefix}global_results' not in st.session_state:
        st.session_state[f'{prefix}global_results'] = {}


# ==================== GESTIÓN DE CONFIGURACIÓN ====================

def update_model_config(model_type: str, config_update: Dict, mode: str = None):
    """
    Actualiza la configuración del modelo en session state.
    
    Args:
        model_type: Tipo de modelo
        config_update: Diccionario con actualizaciones
        mode: Modo específico ('individual', 'global') si aplica
    """
    if mode:
        key = f"{model_type}_unified_{mode}_config"
    else:
        key = f"{model_type}_config"
    
    if key not in st.session_state:
        st.session_state[key] = {}
    
    st.session_state[key].update(config_update)


def get_model_config(model_type: str, mode: str = None) -> Dict:
    """
    Obtiene la configuración actual del modelo.
    
    Args:
        model_type: Tipo de modelo
        mode: Modo específico si aplica
        
    Returns:
        Diccionario con la configuración actual
    """
    if mode:
        key = f"{model_type}_unified_{mode}_config"
    else:
        key = f"{model_type}_config"
    
    return st.session_state.get(key, {})


def clear_model_config(model_type: str, mode: str = None):
    """
    Limpia la configuración del modelo.
    
    Args:
        model_type: Tipo de modelo
        mode: Modo específico si aplica
    """
    if mode:
        key = f"{model_type}_unified_{mode}_config"
    else:
        key = f"{model_type}_config"
    
    if key in st.session_state:
        del st.session_state[key]


# ==================== GESTIÓN DE RESULTADOS ====================

def store_model_results(model_type: str, results: Dict, config_key: str, mode: str = None):
    """
    Almacena los resultados de un modelo en session state.
    
    Args:
        model_type: Tipo de modelo
        results: Diccionario con los resultados
        config_key: Clave única de configuración
        mode: Modo específico si aplica
    """
    if mode:
        key = f"{model_type}_unified_{mode}_results"
    else:
        key = f"{model_type}_analysis_data"
    
    if key not in st.session_state:
        st.session_state[key] = {}
    
    st.session_state[key][config_key] = results


def get_model_results(model_type: str, config_key: str, mode: str = None) -> Optional[Dict]:
    """
    Obtiene los resultados de un modelo específico.
    
    Args:
        model_type: Tipo de modelo
        config_key: Clave única de configuración
        mode: Modo específico si aplica
        
    Returns:
        Diccionario con los resultados o None si no existen
    """
    if mode:
        key = f"{model_type}_unified_{mode}_results"
    else:
        key = f"{model_type}_analysis_data"
    
    return st.session_state.get(key, {}).get(config_key)


def clear_model_results(model_type: str, mode: str = None):
    """
    Limpia todos los resultados almacenados.
    
    Args:
        model_type: Tipo de modelo
        mode: Modo específico si aplica
    """
    if mode:
        key = f"{model_type}_unified_{mode}_results"
    else:
        key = f"{model_type}_analysis_data"
    
    if key in st.session_state:
        st.session_state[key] = {}


# ==================== INTERFAZ DE SELECCIÓN DE VARIABLES ====================

def show_variable_selection_interface(df_master: pd.DataFrame, model_type: str,
                                    key_prefix: str = "") -> List[str]:
    """
    Muestra interfaz estándar de selección de variables.
    
    Args:
        df_master: DataFrame con los datos
        model_type: Tipo de modelo para configuración específica
        key_prefix: Prefijo para las claves de session_state
        
    Returns:
        Lista de variables seleccionadas
    """
    st.subheader("🔧 Selección de Variables")
    
    # Obtener variables disponibles
    available_categories = get_available_features(df_master.columns.tolist())
    
    if not available_categories:
        st.error("No se encontraron variables disponibles en el dataset.")
        return []
    
    # Crear tabs para categorías
    var_tabs = st.tabs(list(available_categories.keys()))
    
    selected_features = []
    
    for i, (category, vars_list) in enumerate(available_categories.items()):
        with var_tabs[i]:
            # Configurar defaults específicos para cada tipo de modelo
            if category == "Variables Temporales":
                if model_type == 'xgboost':
                    # Para XGBoost, preferir variables no cíclicas por defecto
                    default_vars = [var for var in vars_list if not ('sin' in var or 'cos' in var)]
                else:
                    # Para GAM y Bayesian, preferir variables cíclicas
                    default_vars = [var for var in vars_list if 'sin' in var or 'cos' in var or var in ['weekend']]
            else:
                default_vars = vars_list
            
            selected_in_category = st.multiselect(
                f"Variables de {category}",
                vars_list,
                default=default_vars,
                help=f"Selecciona las variables de {category.lower()} para el modelo",
                key=f"{key_prefix}_{category.replace(' ', '_')}_vars"
            )
            
            selected_features.extend(selected_in_category)
    
    # Validar selección
    if not validate_feature_selection(selected_features):
        return []
    
    return selected_features


# ==================== INTERFAZ DE CONFIGURACIÓN ====================

def show_preprocessing_configuration(model_type: str, key_prefix: str = "") -> Dict:
    """
    Muestra interfaz de configuración de preprocesamiento.
    
    Args:
        model_type: Tipo de modelo para defaults específicos
        key_prefix: Prefijo para las claves de session_state
        
    Returns:
        Diccionario con configuración de preprocesamiento
    """
    config = {}
    
    # Obtener defaults según el tipo de modelo
    from ..config import get_model_config
    model_defaults = get_model_config(model_type)
    
    default_outlier = model_defaults.get('default_outlier_method', 'none')
    default_preprocessing = model_defaults.get('default_preprocessing', 'none')
    
    # Método de filtrado de outliers
    outlier_method = st.selectbox(
        "Método de filtrado de outliers",
        options=list(OUTLIER_METHODS.keys()),
        format_func=lambda x: OUTLIER_METHODS[x],
        index=list(OUTLIER_METHODS.keys()).index(default_outlier),
        key=f"{key_prefix}_outlier_method",
        help="Los outliers se eliminan SOLO del conjunto de entrenamiento"
    )
    config['outlier_method'] = outlier_method
    
    # Preprocesamiento temporal
    preprocessing = st.selectbox(
        "Preprocesamiento temporal",
        options=list(PREPROCESSING_OPTIONS.keys()),
        format_func=lambda x: PREPROCESSING_OPTIONS[x],
        index=list(PREPROCESSING_OPTIONS.keys()).index(default_preprocessing),
        key=f"{key_prefix}_preprocessing",
        help="Variables cíclicas (sin/cos) preservan continuidad temporal"
    )
    config['preprocessing'] = preprocessing
    
    return config


def show_individual_configuration(df_master: pd.DataFrame, model_type: str, 
                                key_prefix: str = "") -> Dict:
    """
    Muestra configuración para modelos individuales (por sensor).
    
    Args:
        df_master: DataFrame con los datos
        model_type: Tipo de modelo
        key_prefix: Prefijo para las claves
        
    Returns:
        Diccionario con configuración individual
    """
    config = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de sensor
        sensores = sorted(df_master['id_no2'].unique())
        sensor_seleccionado = st.selectbox(
            "Sensor de NO₂", 
            sensores, 
            index=2 if len(sensores) > 2 else 0,
            key=f"{key_prefix}_sensor",
            help="Sensor para entrenamiento individual"
        )
        config['sensor'] = sensor_seleccionado
        
        # Filtrar por sensor y obtener fechas
        df_sensor = df_master[df_master['id_no2'] == sensor_seleccionado]
        fecha_min = df_sensor["fecha"].min().date()
        fecha_max = df_sensor["fecha"].max().date()
        
        # Fecha de división
        fecha_division = st.date_input(
            "Fecha de división (entrenamiento/prueba)",
            value=pd.to_datetime('2024-01-01').date(),
            min_value=fecha_min,
            max_value=fecha_max,
            help="Los datos anteriores se usan para entrenamiento, posteriores para evaluación",
            key=f"{key_prefix}_split_date"
        )
        config['fecha_division'] = fecha_division
        config['df_sensor'] = df_sensor
    
    with col2:
        # Configuración de preprocesamiento
        preprocessing_config = show_preprocessing_configuration(model_type, key_prefix)
        config.update(preprocessing_config)
    
    return config


def show_global_configuration(df_master: pd.DataFrame, model_type: str,
                            key_prefix: str = "") -> Dict:
    """
    Muestra configuración para modelos globales (multi-sensor).
    
    Args:
        df_master: DataFrame con los datos
        model_type: Tipo de modelo
        key_prefix: Prefijo para las claves
        
    Returns:
        Diccionario con configuración global
    """
    config = {}
    
    st.markdown("### 🌍 Configuración Global Multi-Sensor")
    
    # Configuración de sensores para train/test
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Sensores para Entrenamiento**")
        sensores_disponibles = sorted(df_master['id_no2'].unique())
        sensores_train = st.multiselect(
            "Selecciona sensores para entrenar:",
            sensores_disponibles,
            default=sensores_disponibles[:-2],  # Todos menos los últimos 2
            key=f"{key_prefix}_train_sensors",
            help="Sensores que se usarán para entrenar el modelo global"
        )
        config['sensores_train'] = sensores_train
    
    with col2:
        st.markdown("**🧪 Sensores para Evaluación**")
        sensores_test = st.multiselect(
            "Selecciona sensores para evaluar:",
            sensores_disponibles,
            default=sensores_disponibles[-2:],  # Los últimos 2
            key=f"{key_prefix}_test_sensors",
            help="Sensores que se usarán para evaluar la transferibilidad"
        )
        config['sensores_test'] = sensores_test
    
    # Validaciones
    if not sensores_train or not sensores_test:
        if not sensores_train:
            st.warning("⚠️ Selecciona al menos un sensor para entrenamiento")
        if not sensores_test:
            st.warning("⚠️ Selecciona al menos un sensor para evaluación")
        return config
    
    # Mostrar estadísticas
    show_global_configuration_stats(df_master, sensores_train, sensores_test)
    
    # Configuración adicional
    col1, col2 = st.columns(2)
    with col1:
        preprocessing_config = show_preprocessing_configuration(model_type, key_prefix)
        config.update(preprocessing_config)
    
    return config


def show_global_configuration_stats(df_master: pd.DataFrame, sensores_train: List[str], 
                                  sensores_test: List[str]):
    """
    Muestra estadísticas de configuración global.
    
    Args:
        df_master: DataFrame principal
        sensores_train: Lista de sensores de entrenamiento
        sensores_test: Lista de sensores de evaluación
    """
    df_train = df_master[df_master['id_no2'].isin(sensores_train)]
    df_test = df_master[df_master['id_no2'].isin(sensores_test)]
    
    st.markdown("### 📊 Estadísticas de la Configuración")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sensores entrenamiento", len(sensores_train))
    with col2:
        st.metric("Registros entrenamiento", f"{len(df_train):,}")
    with col3:
        st.metric("Sensores evaluación", len(sensores_test))
    with col4:
        st.metric("Registros evaluación", f"{len(df_test):,}")


# ==================== INTERFAZ DE RESUMEN ====================

def show_configuration_summary(config: Dict, selected_features: List[str], 
                              model_type: str, mode: str = None):
    """
    Muestra resumen de configuración.
    
    Args:
        config: Configuración del modelo
        selected_features: Variables seleccionadas
        model_type: Tipo de modelo
        mode: Modo específico si aplica
    """
    with st.expander("📋 Resumen de Configuración"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Objetivo del Modelo**")
            if 'sensor' in config:
                st.write(f"**Sensor:** {config['sensor']}")
                st.write(f"**Tipo:** Individual")
            else:
                st.write(f"**Sensores Train:** {len(config.get('sensores_train', []))}")
                st.write(f"**Sensores Test:** {len(config.get('sensores_test', []))}")
                st.write(f"**Tipo:** Global")
            st.write(f"**Algoritmo:** {model_type.upper()}")
        
        with col2:
            st.markdown("**⚙️ Preprocesamiento**")
            st.write(f"**Variables:** {len(selected_features)}")
            st.write(f"**Outliers:** {OUTLIER_METHODS[config['outlier_method']]}")
            st.write(f"**Temporal:** {PREPROCESSING_OPTIONS[config['preprocessing']]}")
        
        with col3:
            st.markdown("**📅 División Temporal**")
            if 'fecha_division' in config:
                st.write(f"**Fecha división:** {config['fecha_division']}")
                if 'df_sensor' in config:
                    fecha_min = config['df_sensor']["fecha"].min().date()
                    st.write(f"**Período entreno:** {fecha_min} → {config['fecha_division']}")


# ==================== GESTIÓN DE ESTADO DE ANÁLISIS ====================

def get_analysis_tab_state(model_type: str, mode: str = None) -> int:
    """
    Obtiene el estado actual de la pestaña de análisis.
    
    Args:
        model_type: Tipo de modelo
        mode: Modo específico si aplica
        
    Returns:
        Índice de la pestaña activa
    """
    if mode:
        key = f"{model_type}_unified_{mode}_analysis_tab"
    else:
        key = f"{model_type}_analysis_tab"
    
    return st.session_state.get(key, 0)


def set_analysis_tab_state(model_type: str, tab_index: int, mode: str = None):
    """
    Establece el estado de la pestaña de análisis.
    
    Args:
        model_type: Tipo de modelo
        tab_index: Índice de la pestaña
        mode: Modo específico si aplica
    """
    if mode:
        key = f"{model_type}_unified_{mode}_analysis_tab"
    else:
        key = f"{model_type}_analysis_tab"
    
    st.session_state[key] = tab_index


# ==================== UTILIDADES DE DEPURACIÓN ====================

def debug_session_state(model_type: str, show_full: bool = False):
    """
    Muestra información de depuración del session state.
    
    Args:
        model_type: Tipo de modelo a debuggear
        show_full: Si mostrar todo el contenido o solo claves
    """
    with st.expander("🐛 Debug Session State"):
        relevant_keys = [key for key in st.session_state.keys() if model_type in key]
        
        if show_full:
            for key in relevant_keys:
                st.write(f"**{key}:**")
                st.json(st.session_state[key] if isinstance(st.session_state[key], (dict, list)) 
                       else str(st.session_state[key]))
        else:
            st.write("**Claves relevantes:**")
            for key in relevant_keys:
                value_type = type(st.session_state[key]).__name__
                if isinstance(st.session_state[key], (dict, list)):
                    size = len(st.session_state[key])
                    st.write(f"- `{key}`: {value_type} (size: {size})")
                else:
                    st.write(f"- `{key}`: {value_type}")


def reset_session_state(model_type: str, confirm: bool = False):
    """
    Resetea el session state para un modelo específico.
    
    Args:
        model_type: Tipo de modelo
        confirm: Si requerir confirmación
    """
    if confirm:
        if st.button(f"🗑️ Resetear estado de {model_type.upper()}", type="secondary"):
            keys_to_remove = [key for key in st.session_state.keys() if model_type in key]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success(f"Estado de {model_type.upper()} reseteado correctamente.")
            st.rerun()
    else:
        keys_to_remove = [key for key in st.session_state.keys() if model_type in key]
        for key in keys_to_remove:
            del st.session_state[key] 