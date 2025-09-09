"""
Aplicación principal refactorizada para análisis de contaminación y tráfico en Madrid.

Esta aplicación proporciona múltiples módulos de análisis a través de un sistema de tabs
"""

import streamlit as st
from typing import Dict, Callable
import sys
import os
import pandas as pd

# Configuración de rutas y formato de números
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pd.options.display.float_format = "{:.2f}".format

# Importar configuración centralizada
from src.config import PAGE_CONFIG, TAB_CONFIG

# Importar módulos refactorizados
from src.welcome_page import welcome_page
from src.no2_analysis import generar_analisis_no2
from src.sensor_mapping import generar_mapa_asignaciones
from src.correlations_analysis import analisis_sensores
from src.gam_training import training_page
from src.xgboost_unified import xgboost_unified_page
from src.bayesian_nowcasting import bayesian_nowcasting_page

# ==================== CONFIGURACIÓN DE TABS CON FUNCIONES ====================

TAB_FUNCTIONS = {
    "Inicio": welcome_page,
    "Análisis NO₂": generar_analisis_no2,
    "Mapeo Sensores": generar_mapa_asignaciones,
    "Correlaciones": analisis_sensores,
    "Entrenamiento GAM": training_page,
    "XGBoost Unificado": xgboost_unified_page,
    "Nowcasting Bayesiano (Dropout)": bayesian_nowcasting_page,
}

# ==================== CLASE PRINCIPAL ====================

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disables problematic inspection


class DashboardApp:
    """Clase principal para manejar la aplicación dashboard."""
    
    def __init__(self):
        self._configure_page()
        self._add_custom_css()
        self._initialize_session_state()
    
    def _configure_page(self):
        """Configura la página de Streamlit."""
        st.set_page_config(**PAGE_CONFIG)
    
    def _add_custom_css(self):
        """Añade CSS personalizado para limitar el ancho de la aplicación."""
        st.markdown("""
        <style>
        /* Layout intermedio personalizado con porcentajes responsive */
        .main .block-container {
            max-width: 85%; /* 85% del ancho de pantalla - se adapta a cualquier resolución */
            margin: 0 auto;
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        /* Mejorar el diseño de los tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 0.25rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
        }
        
        /* Estilo para contenedores de información */
        .info-container {
            background-color: #f0f8ff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #1f4e79;
        }
        
        /* Mejorar espaciado */
        .stMarkdown, .stTitle, .stHeader, .stSubheader {
            margin-bottom: 1rem;
        }
        
        /* Responsive design por tamaño de pantalla */
        @media (min-width: 2560px) {
            /* Monitores 4K y superiores - más restrictivo */
            .main .block-container {
                max-width: 75%;
            }
        }
        
        @media (max-width: 1440px) {
            /* Monitores estándar y laptops */
            .main .block-container {
                max-width: 90%;
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
        }
        
        @media (max-width: 768px) {
            /* Tablets y móviles */
            .main .block-container {
                max-width: 95%;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Mostrar opciones de ancho en un expander (comentado por defecto)
        # self._show_width_options()
    
    def _show_width_options(self):
        """Muestra opciones para cambiar el ancho de la aplicación."""
        with st.expander("🔧 Opciones de Ancho Responsive", expanded=False):
            st.markdown("""
            **Opciones disponibles para el ancho de la aplicación (responsive):**
            
            - **Narrow (70%)**: Ideal para lectura enfocada
            - **Medium (85%)**: Equilibrio perfecto (actual)
            - **Large (95%)**: Más espacio para gráficos complejos
            - **Full (100%)**: Utiliza todo el ancho disponible
            
            **Ventajas del diseño responsive:**
            - 🖥️ **1080p**: 85% = ~1632px
            - 🖥️ **1440p**: 85% = ~2176px  
            - 🖥️ **4K**: 85% = ~3264px
            - 📱 **Mobile**: Se adapta automáticamente
            """)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Narrow (70%)"):
                    self._apply_width_percentage("70%")
            with col2:
                if st.button("Medium (85%)"):
                    self._apply_width_percentage("85%")
            with col3:
                if st.button("Large (95%)"):
                    self._apply_width_percentage("95%")
            with col4:
                if st.button("Full (100%)"):
                    self._apply_width_percentage("100%")
    
    def _apply_width_percentage(self, width: str):
        """Aplica un ancho específico en porcentaje mediante CSS dinámico."""
        st.markdown(f"""
        <style>
        .main .block-container {{
            max-width: {width} !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        st.success(f"Ancho cambiado a {width} del ancho de pantalla")
        st.rerun()
    
    def _initialize_session_state(self):
        """Inicializa variables globales de session_state."""
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = list(TAB_CONFIG.keys())[0]
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
    
    def _show_header(self):
        """Muestra el header principal de la aplicación."""
        # Contenedor principal para el header
        with st.container():
            st.title("🌍 Dashboard Madrid - Calidad del Aire y Tráfico")

    
    def _show_tab_description(self, tab_name: str):
        """Muestra la descripción del tab actual."""
        config = TAB_CONFIG.get(tab_name, {})
        if config.get('description'):
            st.markdown(f"""
            <div class="info-container">
                <h4 style="margin: 0; color: #1f4e79;">
                    {config.get('icon', '')} {tab_name}
                </h4>
                <p style="margin: 0.5rem 0 0 0; color: #666;">
                    {config['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def _create_tabs(self) -> Dict:
        """Crea y configura los tabs de la aplicación."""
        tab_names = list(TAB_CONFIG.keys())
        tab_labels = [f"{TAB_CONFIG[name]['icon']} {name}" for name in tab_names]
        
        # Contenedor para los tabs
        with st.container():
            tabs = st.tabs(tab_labels)
        return dict(zip(tab_names, tabs))
    
    def _execute_tab_function(self, tab_name: str, tab_container):
        """Ejecuta la función asociada a un tab específico."""
        config = TAB_CONFIG.get(tab_name)
        
        if not config:
            st.error(f"Configuración no encontrada para el tab: {tab_name}")
            return
        
        function = TAB_FUNCTIONS.get(tab_name)
        
        if not function:
            st.error(f"Función no definida para el tab: {tab_name}")
            return
        
        try:
            with tab_container:
                # Contenedor para el contenido del tab
                with st.container():
                    # Mostrar descripción del tab
                    #self._show_tab_description(tab_name)
                    
                    # Ejecutar función del tab
                    function()
                
        except Exception as e:
            st.error(f"Error al ejecutar {tab_name}: {str(e)}")
            st.exception(e)
    
    def _show_footer(self):
        """Muestra el footer de la aplicación."""
        with st.container():
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
                <p>
                    <strong>Dashboard Madrid - Calidad del Aire y Tráfico</strong><br>
                    Datos proporcionados por el Ayuntamiento de Madrid<br>
                    Desarrollado para análisis de investigación científica
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Método principal para ejecutar la aplicación."""
        try:
            # Contenedor principal de la aplicación
            with st.container():
                # Mostrar header
                self._show_header()
                
                # Crear tabs
                tabs = self._create_tabs()
                
                # Ejecutar funciones de cada tab
                for tab_name, tab_container in tabs.items():
                    self._execute_tab_function(tab_name, tab_container)
                
                # Mostrar footer
                self._show_footer()
            
        except Exception as e:
            st.error("Error crítico en la aplicación")
            st.exception(e)


# ==================== FUNCIONES AUXILIARES ====================

def show_sidebar_info():
    """Muestra información general en el sidebar."""
    with st.sidebar:
        st.markdown("## ℹ️ Información")
        st.markdown("""
        **Navegación:**
        - Usa las pestañas superiores para cambiar entre módulos
        - Cada módulo tiene controles específicos en este panel lateral
        """)
        
        st.markdown("---")


def handle_navigation():
    """Maneja la navegación entre tabs."""
    # Esta función puede expandirse para manejar navegación más compleja
    # Por ejemplo, deep linking, estado persistente entre tabs, etc.
    pass


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal de la aplicación."""
    # Crear y ejecutar la aplicación
    app = DashboardApp()
    
    # Mostrar información general en sidebar
    #show_sidebar_info()
    
    # Manejar navegación
    handle_navigation()
    
    # Ejecutar aplicación principal
    app.run()


if __name__ == "__main__":
    main() 