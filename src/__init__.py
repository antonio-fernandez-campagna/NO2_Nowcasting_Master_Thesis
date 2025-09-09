"""
Paquete principal para la aplicación de análisis de NO₂ en Madrid.

Este paquete contiene todos los módulos para el análisis integrado
de datos de contaminación atmosférica, tráfico y meteorología.
"""

__version__ = "2.0.0"
__author__ = "Antonio Fernández"
__email__ = ""
__description__ = "Dashboard de Análisis de NO₂ en Madrid"

# Importaciones principales del paquete
from .config import *
from .utils import *

# Información del paquete
__all__ = [
    "config",
    "utils",
    "app", 
    "welcome_page",
    "no2_analysis",
    "sensor_mapping",
    "correlations_analysis", 
    "gam_training",
    "xgboost_training",
    "bayesian_nowcasting"
] 