"""
Configuración centralizada para todos los modelos de predicción de NO2.

Este módulo contiene todas las constantes, configuraciones y metadatos
utilizados por los diferentes algoritmos de machine learning.
"""

from typing import Dict, List
import streamlit as st

# ==================== MÉTODOS DE FILTRADO DE OUTLIERS ====================

OUTLIER_METHODS = {
    'iqr': 'Rango Intercuartílico (IQR)',
    'zscore': 'Z-Score (Desviación Estándar)',
    'quantiles': 'Percentiles Extremos',
    'none': 'Sin filtrado'
}

# ==================== OPCIONES DE PREPROCESAMIENTO ====================

PREPROCESSING_OPTIONS = {
    'sin_cos': 'Variables Cíclicas (Sin/Cos)',
    'none': 'Sin preprocesamiento'
}

# ==================== CATEGORÍAS DE VARIABLES ====================

VARIABLE_CATEGORIES = {
    "Variables Temporales": [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
        'weekend', 'season_sin', 'season_cos', 'hour', 'month', 'day_of_week', 'day_of_year'
    ],
    "Variables de Tráfico": [
        'intensidad', 'carga', 'ocupacion',
        # Lags de tráfico
        'intensidad_lag1', 'intensidad_lag2', 'intensidad_lag3', 'intensidad_lag4', 'intensidad_lag6', 'intensidad_lag8',
        'ocupacion_lag1', 'ocupacion_lag2', 'ocupacion_lag3', 'ocupacion_lag4', 'ocupacion_lag6', 'ocupacion_lag8',
        'carga_lag1', 'carga_lag2', 'carga_lag3', 'carga_lag4', 'carga_lag6', 'carga_lag8'
    ],
    "Variables Meteorológicas": [
        'd2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction_deg',
        'wind_dir_sin', 'wind_dir_cos'
    ],
    "Variables Meteorológicas - Medias Móviles": [
        # SMA para wind_speed
        'wind_speed_ma3', 'wind_speed_ma6', 'wind_speed_ma24',
        # SMA para temperatura
        't2m_ma6', 't2m_ma24',
        # SMA para punto de rocío
        'd2m_ma6', 'd2m_ma24',
        # SMA para presión
        'sp_ma6', 'sp_ma24', 'sp_ma72',
        # SMA para componentes de viento
        'u10_ma6', 'u10_ma24', 'v10_ma6', 'v10_ma24'
    ],
    "Variables Meteorológicas - Medias Exponenciales": [
        # EWM
        'wind_speed_ewm3', 't2m_ewm6', 'd2m_ewm6', 'sp_ewm12', 'u10_ewm6', 'v10_ewm6'
    ],
    "Variables Meteorológicas - Acumuladas": [
        # Sumas acumuladas
        'ssr_sum24', 'ssrd_sum24', 'tp_sum6', 'tp_sum24'
    ],
    "Variables de Dirección de Viento Suavizadas": [
        # Dirección de viento suavizada
        'wind_dir_sin_ma6', 'wind_dir_sin_ma24', 'wind_dir_cos_ma6', 'wind_dir_cos_ma24',
        'wind_dir_deg_ma6', 'wind_dir_deg_ma24'
    ]
}

# ==================== METADATOS DE VARIABLES ====================

VARIABLE_METADATA = {
    # Variables meteorológicas básicas
    'd2m': {'name': 'Punto de Rocío', 'unit': '°C', 'typical_range': (-10, 30)},
    't2m': {'name': 'Temperatura', 'unit': '°C', 'typical_range': (-5, 40)},
    'sp': {'name': 'Presión Superficial', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'tp':  {'name': 'Precipitación Total',      'unit': 'mm',     'typical_range': (0, 50)},
    'ssr':  {'name': 'Radiación Solar Neta',        'unit': 'kWh/m²', 'typical_range': (0, 1)},
    'ssrd': {'name': 'Radiación Solar Descendente', 'unit': 'kWh/m²', 'typical_range': (0, 1)},
    'u10':  {'name': 'Viento U 10m',                 'unit': 'km/h',   'typical_range': (-180, 180)},
    'v10':  {'name': 'Viento V 10m',                 'unit': 'km/h',   'typical_range': (-180, 180)},
    'wind_speed': {'name': 'Velocidad del Viento',  'unit': 'km/h',   'typical_range': (0, 150)},
    
    
    # Variables de tráfico básicas
    'intensidad': {'name': 'Intensidad de Tráfico', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'carga': {'name': 'Carga de Tráfico', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion': {'name': 'Ocupación Vial', 'unit': '%', 'typical_range': (0, 100)},
    'vmed': {'name': 'Velocidad Media', 'unit': 'km/h', 'typical_range': (0, 100)},
    
    # Variables de tráfico con lags
    'intensidad_lag1': {'name': 'Intensidad Tráfico (t-1)', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'intensidad_lag2': {'name': 'Intensidad Tráfico (t-2)', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'intensidad_lag3': {'name': 'Intensidad Tráfico (t-3)', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'intensidad_lag4': {'name': 'Intensidad Tráfico (t-4)', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'intensidad_lag6': {'name': 'Intensidad Tráfico (t-6)', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'intensidad_lag8': {'name': 'Intensidad Tráfico (t-8)', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'ocupacion_lag1': {'name': 'Ocupación Vial (t-1)', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion_lag2': {'name': 'Ocupación Vial (t-2)', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion_lag3': {'name': 'Ocupación Vial (t-3)', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion_lag4': {'name': 'Ocupación Vial (t-4)', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion_lag6': {'name': 'Ocupación Vial (t-6)', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion_lag8': {'name': 'Ocupación Vial (t-8)', 'unit': '%', 'typical_range': (0, 100)},
    'carga_lag1': {'name': 'Carga Tráfico (t-1)', 'unit': '%', 'typical_range': (0, 100)},
    'carga_lag2': {'name': 'Carga Tráfico (t-2)', 'unit': '%', 'typical_range': (0, 100)},
    'carga_lag3': {'name': 'Carga Tráfico (t-3)', 'unit': '%', 'typical_range': (0, 100)},
    'carga_lag4': {'name': 'Carga Tráfico (t-4)', 'unit': '%', 'typical_range': (0, 100)},
    'carga_lag6': {'name': 'Carga Tráfico (t-6)', 'unit': '%', 'typical_range': (0, 100)},
    'carga_lag8': {'name': 'Carga Tráfico (t-8)', 'unit': '%', 'typical_range': (0, 100)},
    
    # Variables meteorológicas - Medias móviles (SMA)
    'wind_speed_ma3': {'name': 'Velocidad Viento MA-3h', 'unit': 'km/h', 'typical_range': (0, 100)},
    'wind_speed_ma6': {'name': 'Velocidad Viento MA-6h', 'unit': 'km/h', 'typical_range': (0, 100)},
    'wind_speed_ma24': {'name': 'Velocidad Viento MA-24h', 'unit': 'km/h', 'typical_range': (0, 100)},
    't2m_ma6': {'name': 'Temperatura MA-6h', 'unit': '°C', 'typical_range': (-5, 40)},
    't2m_ma24': {'name': 'Temperatura MA-24h', 'unit': '°C', 'typical_range': (-5, 40)},
    'd2m_ma6': {'name': 'Punto Rocío MA-6h', 'unit': '°C', 'typical_range': (-10, 30)},
    'd2m_ma24': {'name': 'Punto Rocío MA-24h', 'unit': '°C', 'typical_range': (-10, 30)},
    'sp_ma6': {'name': 'Presión MA-6h', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'sp_ma24': {'name': 'Presión MA-24h', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'sp_ma72': {'name': 'Presión MA-72h', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'u10_ma6': {'name': 'Viento U MA-6h', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'u10_ma24': {'name': 'Viento U MA-24h', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'v10_ma6': {'name': 'Viento V MA-6h', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'v10_ma24': {'name': 'Viento V MA-24h', 'unit': 'km/h', 'typical_range': (-50, 50)},
    
    # Variables meteorológicas - Medias exponenciales (EWM)
    'wind_speed_ewm3': {'name': 'Velocidad Viento EWM-3h', 'unit': 'km/h', 'typical_range': (0, 100)},
    't2m_ewm6': {'name': 'Temperatura EWM-6h', 'unit': '°C', 'typical_range': (-5, 40)},
    'd2m_ewm6': {'name': 'Punto Rocío EWM-6h', 'unit': '°C', 'typical_range': (-10, 30)},
    'sp_ewm12': {'name': 'Presión EWM-12h', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'u10_ewm6': {'name': 'Viento U EWM-6h', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'v10_ewm6': {'name': 'Viento V EWM-6h', 'unit': 'km/h', 'typical_range': (-50, 50)},
    
    # Variables meteorológicas - Acumuladas
    'ssr_sum24':  {'name': 'Radiación Neta Sum-24h',  'unit': 'kWh/m²', 'typical_range': (0, 24)},
    'ssrd_sum24': {'name': 'Radiación Directa Sum-24h', 'unit': 'kWh/m²', 'typical_range': (0, 24)},
    'tp_sum6':   {'name': 'Precipitación Sum-6h',     'unit': 'mm',     'typical_range': (0, 300)},
    'tp_sum24':  {'name': 'Precipitación Sum-24h',    'unit': 'mm',     'typical_range': (0, 120)},
    
    # Variables de dirección de viento suavizadas
    'wind_dir_sin_ma6': {'name': 'Dir. Viento Sin MA-6h', 'unit': 'sin(°)', 'typical_range': (-1, 1)},
    'wind_dir_sin_ma24': {'name': 'Dir. Viento Sin MA-24h', 'unit': 'sin(°)', 'typical_range': (-1, 1)},
    'wind_dir_cos_ma6': {'name': 'Dir. Viento Cos MA-6h', 'unit': 'cos(°)', 'typical_range': (-1, 1)},
    'wind_dir_cos_ma24': {'name': 'Dir. Viento Cos MA-24h', 'unit': 'cos(°)', 'typical_range': (-1, 1)},
    'wind_dir_deg_ma6': {'name': 'Dir. Viento MA-6h', 'unit': '°', 'typical_range': (0, 360)},
    'wind_dir_deg_ma24': {'name': 'Dir. Viento MA-24h', 'unit': '°', 'typical_range': (0, 360)},
    
    # ───────────── Dirección del viento ─────────────
    'wind_direction_deg': {'name': 'Dirección del Viento',      'unit': '°',      'typical_range': (0, 360)},
    'wind_dir_sin':        {'name': 'Dirección del Viento (Sin)', 'unit': 'sin(°)', 'typical_range': (-1, 1)},
    'wind_dir_cos':        {'name': 'Dirección del Viento (Cos)', 'unit': 'cos(°)', 'typical_range': (-1, 1)},
}

# ==================== COLUMNAS PARA DETECCIÓN DE OUTLIERS ====================

COLUMNS_FOR_OUTLIERS = [
    'no2_value',
    # Variables de tráfico básicas
    'intensidad', 'carga', 'ocupacion', 'vmed',
    # Variables meteorológicas básicas
    'd2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed',
    # Se podrían añadir variables derivadas si se detectan outliers problemáticos
    # Las variables lag y suavizadas suelen tener menos outliers por su naturaleza
]

# ==================== CONFIGURACIONES DE MODELOS ====================

MODEL_CONFIGS = {
    'xgboost': {
        'name': 'XGBoost',
        'description': 'eXtreme Gradient Boosting',
        'icon': '🚀',
        'color': '#FF6B6B',
        'default_outlier_method': 'none',  # XGBoost maneja outliers naturalmente
        'default_preprocessing': 'none',   # XGBoost puede manejar variables temporales directamente
        'supports_feature_importance': True,
        'supports_early_stopping': True
    },
    'gam': {
        'name': 'GAM',
        'description': 'Generalized Additive Models',
        'icon': '📈',
        'color': '#4ECDC4',
        'default_outlier_method': 'iqr',
        'default_preprocessing': 'sin_cos',
        'supports_feature_importance': False,
        'supports_early_stopping': False
    },
    'bayesian': {
        'name': 'Bayesian NN',
        'description': 'Bayesian Neural Networks',
        'icon': '🧠',
        'color': '#45B7D1',
        'default_outlier_method': 'iqr',
        'default_preprocessing': 'sin_cos',
        'supports_feature_importance': False,
        'supports_early_stopping': True,
        'supports_uncertainty': True
    }
}

# ==================== CONFIGURACIONES DE STREAMLIT ====================

STREAMLIT_CONFIG = {
    'page_title': 'Predicción de NO₂',
    'page_icon': '🌍',
    'layout': 'centered',
    'initial_sidebar_state': 'expanded'
}

# ==================== RUTAS DE ARCHIVOS ====================

FILE_PATHS = {
    'data': 'data/super_processed/7_4_no2_with_traffic_and_1meteo_and_1trafic_id.parquet',
    'models_dir': 'data/models',
    'figures_dir': 'data/figures'
}

# ==================== CONFIGURACIONES DE ENTRENAMIENTO ====================

TRAINING_CONFIGS = {
    'xgboost': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 50
    },
    'bayesian': {
        'epochs': 150,
        'batch_size': 64,
        'learning_rate': 0.01,
        'early_stopping_patience': 25,
        'reduce_lr_patience': 12
    },
    'gam': {
        'n_splines': 25,
        'spline_order': 3,
        'reg_lambda': 0.6
    }
}

# ==================== FUNCIONES DE UTILIDAD PARA CONFIGURACIÓN ====================

def get_model_config(model_type: str) -> Dict:
    """Obtiene la configuración para un tipo de modelo específico."""
    return MODEL_CONFIGS.get(model_type, {})

def get_training_config(model_type: str) -> Dict:
    """Obtiene la configuración de entrenamiento para un tipo de modelo."""
    return TRAINING_CONFIGS.get(model_type, {})

def get_variable_categories() -> Dict[str, List[str]]:
    """Obtiene las categorías de variables disponibles."""
    return VARIABLE_CATEGORIES.copy()

def get_available_features(df_columns: List[str]) -> Dict[str, List[str]]:
    """
    Filtra las variables disponibles basándose en las columnas del DataFrame.
    
    Args:
        df_columns: Lista de columnas disponibles en el DataFrame
        
    Returns:
        Diccionario con categorías y variables disponibles
    """
    available_categories = {}
    
    for category, vars_list in VARIABLE_CATEGORIES.items():
        available_vars = []
        for var in vars_list:
            if var in df_columns or 'sin' in var or 'cos' in var:
                available_vars.append(var)
        
        if available_vars:  # Solo incluir categorías con variables disponibles
            available_categories[category] = available_vars
    
    return available_categories

def validate_feature_selection(selected_features: List[str]) -> bool:
    """Valida que la selección de features sea válida."""
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable para continuar.")
        return False
    
    # Validar que no haya conflictos entre variables cíclicas y no cíclicas
    temporal_vars = set(selected_features) & set(VARIABLE_CATEGORIES["Variables Temporales"])
    
    cyclical_vars = [var for var in temporal_vars if 'sin' in var or 'cos' in var]
    non_cyclical_vars = [var for var in temporal_vars if 'sin' not in var and 'cos' not in var]
    
    # Verificar si hay variables temporales base que también tienen versión cíclica seleccionada
    base_vars = {'hour', 'month', 'day_of_week', 'day_of_year'}
    cyclical_base_vars = set()
    
    for cyclical_var in cyclical_vars:
        for base_var in base_vars:
            if base_var in cyclical_var:
                cyclical_base_vars.add(base_var)
    
    conflicting_vars = set(non_cyclical_vars) & cyclical_base_vars
    
    if conflicting_vars:
        st.warning(f"⚠️ Variables conflictivas detectadas: {conflicting_vars}. "
                   f"No selecciones tanto la versión cíclica como la no cíclica de la misma variable temporal.")
        return False
    
    return True

def get_session_state_key(prefix: str, **kwargs) -> str:
    """
    Genera una clave única para session_state basada en parámetros.
    
    Args:
        prefix: Prefijo para la clave
        **kwargs: Parámetros adicionales para la clave
        
    Returns:
        Clave única como string
    """
    key_parts = [prefix]
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (list, tuple)):
            key_parts.append(f"{key}_{len(value)}")
        else:
            key_parts.append(f"{key}_{value}")
    
    return "_".join(str(part) for part in key_parts)

# ==================== CONFIGURACIONES DE STREAMLIT (ACTUALIZADO) ====================

PAGE_CONFIG = {
    'page_title': 'Dashboard Madrid NO₂',
    'page_icon': '🌍',
    'layout': 'wide',
    'menu_items': {
        'Get Help': 'https://github.com/tu-repo/issues',
        'Report a bug': 'https://github.com/tu-repo/issues',
        'About': "Dashboard de análisis de NO₂ en Madrid - Tesis de Maestría"
    }
}

# ==================== CONFIGURACIÓN DE TABS ====================

TAB_CONFIG = {
    "Inicio": {
        'icon': '🏠',
        'description': 'Página principal con información del proyecto y guía de navegación',
        'requires_data': False
    },
    "Análisis NO₂": {
        'icon': '🌫️',
        'description': 'Análisis exploratorio de datos de concentración de NO₂',
        'requires_data': True
    },
    "Mapeo Sensores": {
        'icon': '🗺️',
        'description': 'Visualización geoespacial de sensores de calidad del aire y tráfico',
        'requires_data': True
    },
    "Correlaciones": {
        'icon': '🔗',
        'description': 'Análisis de correlaciones entre variables meteorológicas, tráfico y NO₂',
        'requires_data': True
    },
    "Entrenamiento GAM": {
        'icon': '📈',
        'description': 'Modelos aditivos generalizados para predicción de NO₂',
        'requires_data': True
    },
    "XGBoost Unificado": {
        'icon': '🚀',
        'description': 'Entrenamiento y análisis de múltiples modelos XGBoost para todos los sensores de forma unificada',
        'requires_data': True
    },
    "Nowcasting Bayesiano (Dropout)": {
        'icon': '🧠',
        'description': 'Predicción a corto plazo (nowcasting) usando redes neuronales con dropout para estimar la incertidumbre del modelo',
        'requires_data': True
    }
}

# ==================== RUTAS DE DATOS ====================

DATA_PATHS = {
    'NO2_DATA': '../data/super_processed/7_4_no2_with_traffic_and_1meteo_and_1trafic_id.parquet',
    'TRAFFIC_NO2_DATA': '../data/super_processed/7_4_no2_with_traffic_and_1meteo_and_1trafic_id.parquet',
    'SENSOR_MAPPING': '../data/super_processed/4_no2_to_traffic_sensor_mapping.csv'
}

# ==================== CONFIGURACIÓN DE CACHE ====================

CACHE_CONFIG = {
    'ttl': 3600,  # Time to live en segundos (1 hora)
    'show_spinner': True,
    'persist': 'disk'
}

# ==================== CONFIGURACIÓN DE ESTILO ====================

STYLE_CONFIG = {
    'primary_color': '#1f77b4',
    'secondary_color': '#ff7f0e',
    'success_color': '#2ca02c',
    'warning_color': '#ff9800',
    'error_color': '#d32f2f',
    'info_color': '#2196f3',
    'background_color': '#f5f5f5'
}

# ==================== CONFIGURACIÓN DE VARIABLES PARA VISUALIZACIÓN ====================

TRAFFIC_PLOT_VARIABLES = {
    'intensidad': {'range': (0, 1500), 'title': 'Intensidad de Tráfico', 'unit': 'veh/h', 'color': 'red'},
    'carga':      {'range': (0, 100),  'title': 'Carga de Tráfico',      'unit': '%',     'color': 'orange'},
    'ocupacion':  {'range': (0, 100),  'title': 'Ocupación Vial',        'unit': '%',     'color': 'purple'},
    'vmed':       {'range': (0, 100),  'title': 'Velocidad Media',       'unit': 'km/h',  'color': 'blue'}
}

METEO_PLOT_VARIABLES = {
    't2m':      {'range': (-5, 40),   'title': 'Temperatura',               'unit': '°C',      'color': 'red'},
    'd2m':      {'range': (-10, 30),  'title': 'Punto de Rocío',            'unit': '°C',      'color': 'blue'},
    'sp':       {'range': (980, 1030),'title': 'Presión Superficial',       'unit': 'hPa',     'color': 'green'},
    'tp':       {'range': (0, 20),    'title': 'Precipitación',             'unit': 'mm',      'color': 'cyan'},
    'u10':      {'range': (-180, 180),'title': 'Componente U del Viento',   'unit': 'km/h',    'color': 'orange'},
    'v10':      {'range': (-180, 180),'title': 'Componente V del Viento',   'unit': 'km/h',    'color': 'purple'},
    'ssrd':     {'range': (0, 1.2),   'title': 'Radiación Solar',           'unit': 'kWh/m²',  'color': 'yellow'},
    'ssr':      {'range': (0, 1.2),   'title': 'Radiación Neta',            'unit': 'kWh/m²',  'color': 'gold'},
    'wind_speed': {'range': (0, 150), 'title': 'Velocidad del Viento',      'unit': 'km/h',    'color': 'black'}
}


# ==================== FUNCIONES PARA GENERAR CONFIGURACIONES AUTOMÁTICAS ====================

def get_variable_plot_config(variable: str, default_color: str = 'blue') -> Dict:
    """
    Genera configuración de plotting para una variable específica.
    
    Args:
        variable: Nombre de la variable
        default_color: Color por defecto si no está definido
        
    Returns:
        Diccionario con configuración de plotting
    """
    if variable in VARIABLE_METADATA:
        metadata = VARIABLE_METADATA[variable]
        return {
            'range': metadata.get('typical_range', (0, 100)),
            'title': metadata.get('name', variable),
            'unit': metadata.get('unit', ''),
            'color': default_color
        }
    else:
        return {
            'range': (0, 100),
            'title': variable,
            'unit': '',
            'color': default_color
        }

def get_available_traffic_variables(feature_names: List[str]) -> Dict:
    """
    Obtiene todas las variables de tráfico disponibles con su configuración de plotting.
    
    Args:
        feature_names: Lista de nombres de features disponibles
        
    Returns:
        Diccionario con variables de tráfico y su configuración
    """
    traffic_vars = {}
    color_cycle = ['red', 'orange', 'purple', 'blue', 'green', 'cyan', 'magenta', 'brown']
    
    # Primero agregar las variables predefinidas
    for var, config in TRAFFIC_PLOT_VARIABLES.items():
        if var in feature_names:
            traffic_vars[var] = config
    
    # Luego agregar otras variables de tráfico disponibles
    traffic_categories = VARIABLE_CATEGORIES.get("Variables de Tráfico", [])
    color_idx = len(traffic_vars)
    
    for var in traffic_categories:
        if var in feature_names and var not in traffic_vars:
            color = color_cycle[color_idx % len(color_cycle)]
            traffic_vars[var] = get_variable_plot_config(var, color)
            color_idx += 1
    
    return traffic_vars

def get_available_meteo_variables(feature_names: List[str]) -> Dict:
    """
    Obtiene todas las variables meteorológicas disponibles con su configuración de plotting.
    
    Args:
        feature_names: Lista de nombres de features disponibles
        
    Returns:
        Diccionario con variables meteorológicas y su configuración
    """
    meteo_vars = {}
    color_cycle = ['red', 'blue', 'green', 'cyan', 'orange', 'purple', 'yellow', 'gold', 'pink', 'brown']
    
    # Primero agregar las variables predefinidas
    for var, config in METEO_PLOT_VARIABLES.items():
        if var in feature_names:
            meteo_vars[var] = config
    
    # Luego agregar otras variables meteorológicas disponibles
    meteo_categories = [
        "Variables Meteorológicas",
        "Variables Meteorológicas - Medias Móviles", 
        "Variables Meteorológicas - Medias Exponenciales",
        "Variables Meteorológicas - Acumuladas",
        "Variables de Dirección de Viento Suavizadas"
    ]
    
    color_idx = len(meteo_vars)
    
    for category in meteo_categories:
        category_vars = VARIABLE_CATEGORIES.get(category, [])
        for var in category_vars:
            if var in feature_names and var not in meteo_vars:
                # Skip cyclical variables (sin/cos) as they need special handling
                if not any(temp in var for temp in ['sin', 'cos']):
                    color = color_cycle[color_idx % len(color_cycle)]
                    meteo_vars[var] = get_variable_plot_config(var, color)
                    color_idx += 1
    
    return meteo_vars
    return len(traffic_vars) > 0 and len(meteo_vars) > 0

# ==================== CONFIGURACIÓN DE MÉTRICAS ====================

METRICS_CONFIG = {
    'no2_limits': {
        'who_annual': 10,      # µg/m³ - Límite anual OMS
        'who_daily': 25,       # µg/m³ - Límite diario OMS
        'eu_annual': 40,       # µg/m³ - Límite anual UE
        'eu_hourly': 200       # µg/m³ - Límite horario UE
    },
    'air_quality_categories': {
        'buena': {'min': 0, 'max': 40, 'color': '#00e676'},
        'moderada': {'min': 41, 'max': 80, 'color': '#ffeb3b'},
        'mala': {'min': 81, 'max': 120, 'color': '#ff9800'},
        'muy_mala': {'min': 121, 'max': 200, 'color': '#f44336'},
        'extrema': {'min': 201, 'max': float('inf'), 'color': '#9c27b0'}
    }
}