"""
Utilidades de visualización para modelos de predicción de NO2.

Este módulo contiene todas las funciones comunes de visualización, gráficos
y análisis utilizadas por los diferentes algoritmos de machine learning.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
import warnings
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib para mejor presentación
plt.style.use('default')
sns.set_palette("husl")


# ==================== MÉTRICAS DEL MODELO ====================

def show_model_metrics(metrics: Dict, title: str = "Métricas del Modelo"):
    """
    Muestra las métricas del modelo en formato de tarjetas.
    
    Args:
        metrics: Diccionario con métricas (debe incluir 'rmse', 'r2', 'mae')
        title: Título para la sección de métricas
    """
    st.subheader(title)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="RMSE",
            value=f"{metrics['rmse']:.2f} µg/m³",
            help="Error cuadrático medio (menor es mejor)"
        )
    
    with col2:
        st.metric(
            label="R² Score",
            value=f"{metrics['r2']:.3f}",
            help="Coeficiente de determinación (cercano a 1 es mejor)"
        )
    
    with col3:
        st.metric(
            label="MAE",
            value=f"{metrics['mae']:.2f} µg/m³",
            help="Error absoluto medio (menor es mejor)"
        )


def show_additional_metrics(metrics: Dict):
    """
    Muestra métricas adicionales si están disponibles.
    
    Args:
        metrics: Diccionario con métricas adicionales
    """
    additional_metrics = []
    
    if 'mape' in metrics:
        additional_metrics.append(("MAPE", f"{metrics['mape']:.1f}%", "Error porcentual medio"))
    if 'bias' in metrics:
        additional_metrics.append(("Sesgo", f"{metrics['bias']:.3f} µg/m³", "Sesgo sistemático"))
    if 'std_residuals' in metrics:
        additional_metrics.append(("Std Residuos", f"{metrics['std_residuals']:.3f}", "Desviación estándar de residuos"))
    
    if additional_metrics:
        st.subheader("📊 Métricas Adicionales")
        cols = st.columns(len(additional_metrics))
        
        for i, (label, value, help_text) in enumerate(additional_metrics):
            with cols[i]:
                st.metric(label=label, value=value, help=help_text)


# ==================== ANÁLISIS DE RESIDUOS ====================

def show_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Análisis de Residuos"):
    """
    Muestra análisis completo de residuos.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        title: Título para el análisis
    """
    residuals = y_true - y_pred
    
    st.subheader(title)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de residuos
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, ax=ax, bins=30)
        ax.set_title('Distribución de Residuos')
        ax.set_xlabel('Residuo (Real - Predicción) [µg/m³]')
        ax.set_ylabel('Frecuencia')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Error Cero')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Q-Q Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            sm.qqplot(residuals, line='45', ax=ax, fit=True)
            ax.set_title('Q-Q Plot de Residuos')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            # Fallback a scatter plot simple si statsmodels falla
            from scipy import stats
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)
            ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6)
            ax.plot(theoretical_quantiles, theoretical_quantiles, 'r--', alpha=0.7)
            ax.set_title('Q-Q Plot de Residuos')
            ax.set_xlabel('Cuantiles Teóricos')
            ax.set_ylabel('Cuantiles de la Muestra')
            ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    # Estadísticas de residuos
    st.markdown("### 📊 Estadísticas de Residuos")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Media", f"{residuals.mean():.3f}")
    with col2:
        st.metric("Desv. Estándar", f"{residuals.std():.3f}")
    with col3:
        st.metric("Sesgo", f"{residuals.skew():.3f}")
    with col4:
        st.metric("Curtosis", f"{residuals.kurtosis():.3f}")


def show_residuals_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Muestra gráfico de residuos vs valores predichos.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    """
    residuals = y_true - y_pred
    
    st.subheader("📊 Residuos vs Predicciones")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, s=20)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Error Cero')
    ax.set_xlabel('Valores Predichos (µg/m³)')
    ax.set_ylabel('Residuos (µg/m³)')
    ax.set_title('Residuos vs Valores Predichos')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()


# ==================== ANÁLISIS TEMPORAL ====================

def show_temporal_predictions(test_df: pd.DataFrame, y_pred: np.ndarray, 
                            title: str = "Predicciones vs Valores Reales",
                            key_prefix: str = "temporal"):
    """
    Muestra gráficos temporales de predicciones vs valores reales.
    
    Args:
        test_df: DataFrame de prueba con fechas
        y_pred: Predicciones del modelo
        title: Título para los gráficos
        key_prefix: Prefijo para las claves de Streamlit
    """
    df_plot = test_df[['fecha', 'no2_value']].copy()
    df_plot['Predicción'] = y_pred
    df_plot = df_plot.set_index('fecha')
    
    st.subheader(f"📈 {title}")
    
    # Controles para zoom temporal
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Rango de fechas para visualizar:",
            value=(df_plot.index.min().date(), df_plot.index.max().date()),
            min_value=df_plot.index.min().date(),
            max_value=df_plot.index.max().date(),
            key=f"{key_prefix}_predictions_date_range"
        )
    
    with col2:
        granularity = st.selectbox(
            "Granularidad:",
            options=['Horaria', 'Media Diaria', 'Media Semanal'],
            index=0,
            key=f"{key_prefix}_predictions_granularity"
        )
    
    # Filtrar por fechas
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
    df_filtered = df_plot[(df_plot.index >= start_date) & (df_plot.index < end_date)]
    
    if df_filtered.empty:
        st.warning("No hay datos en el rango seleccionado.")
        return
    
    # Aplicar granularidad
    if granularity == 'Media Diaria':
        df_agg = df_filtered.resample('D').mean()
        title_suffix = '(Media Diaria)'
    elif granularity == 'Media Semanal':
        df_agg = df_filtered.resample('W-MON').mean()
        title_suffix = '(Media Semanal)'
    else:
        df_agg = df_filtered
        title_suffix = '(Horario)'
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_agg.index, df_agg['no2_value'], label='Valor Real', alpha=0.8, linewidth=1.5)
    ax.plot(df_agg.index, df_agg['Predicción'], label='Predicción', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Formatear eje X
    if granularity != 'Horaria':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    fig.autofmt_xdate()
    ax.set_title(f'{title} {title_suffix}')
    ax.set_ylabel('Concentración NO₂ (µg/m³)')
    ax.set_xlabel('Fecha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()


def show_residuals_over_time(test_df: pd.DataFrame, y_pred: np.ndarray,
                           title: str = "Análisis Temporal de Errores",
                           key_prefix: str = "temporal"):
    """
    Muestra residuos a lo largo del tiempo.
    
    Args:
        test_df: DataFrame de prueba con fechas
        y_pred: Predicciones del modelo
        title: Título para el análisis
        key_prefix: Prefijo para las claves de Streamlit
    """
    df_plot = test_df[['fecha', 'no2_value']].copy()
    df_plot['Residuos'] = df_plot['no2_value'] - y_pred
    df_plot = df_plot.set_index('fecha')
    
    st.subheader(f"📉 {title}")
    
    # Controles
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Rango de fechas para errores:",
            value=(df_plot.index.min().date(), df_plot.index.max().date()),
            min_value=df_plot.index.min().date(),
            max_value=df_plot.index.max().date(),
            key=f"{key_prefix}_residuals_date_range"
        )
    
    with col2:
        granularity = st.selectbox(
            "Granularidad de errores:",
            options=['Horaria', 'Media Diaria', 'MAE Diario', 'Media Semanal'],
            index=0,
            key=f"{key_prefix}_residuals_granularity"
        )
    
    # Filtrar datos
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
    df_filtered = df_plot[(df_plot.index >= start_date) & (df_plot.index < end_date)]
    
    if df_filtered.empty:
        st.warning("No hay datos de residuos en el rango seleccionado.")
        return
    
    # Aplicar granularidad
    if granularity == 'Media Diaria':
        df_agg = df_filtered.resample('D').mean()
        y_label = 'Residuo Medio (µg/m³)'
        title_suffix = 'Residuos Medios Diarios'
    elif granularity == 'MAE Diario':
        df_agg = df_filtered.resample('D')['Residuos'].apply(lambda x: x.abs().mean()).to_frame()
        df_agg.columns = ['Residuos']
        y_label = 'MAE Diario (µg/m³)'
        title_suffix = 'Error Absoluto Medio Diario'
    elif granularity == 'Media Semanal':
        df_agg = df_filtered.resample('W-MON').mean()
        y_label = 'Residuo Medio (µg/m³)'
        title_suffix = 'Residuos Medios Semanales'
    else:
        df_agg = df_filtered
        y_label = 'Residuo (µg/m³)'
        title_suffix = 'Residuos Horarios'
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_agg.index, df_agg['Residuos'], alpha=0.9, linewidth=1.5)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Error Cero')
    
    # Formatear eje X
    if granularity != 'Horaria':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    fig.autofmt_xdate()
    ax.set_title(title_suffix)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Fecha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()


# ==================== ANÁLISIS DE CORRELACIÓN ====================

def show_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                          title: str = "Predicción vs Real"):
    """
    Muestra scatter plot de predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        title: Título del gráfico
    """
    st.subheader(f"🎯 {title}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Línea ideal (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Predicción Perfecta')
    
    ax.set_xlabel('Valor Real (µg/m³)')
    ax.set_ylabel('Predicción (µg/m³)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Añadir métricas al gráfico
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    st.pyplot(fig)
    plt.close()


# ==================== ANÁLISIS POR HORA DEL DÍA ====================

def show_hourly_patterns(test_df: pd.DataFrame, y_pred: np.ndarray,
                        title: str = "Patrones Horarios"):
    """
    Muestra análisis de patrones por hora del día.
    
    Args:
        test_df: DataFrame de prueba con fechas
        y_pred: Predicciones del modelo
        title: Título del análisis
    """
    if len(test_df) < 24:  # Solo si tenemos suficientes datos
        return
    
    st.subheader(f"🕐 {title}")
    
    df_hourly = test_df.copy()
    df_hourly['hour'] = df_hourly['fecha'].dt.hour
    df_hourly['prediction'] = y_pred
    df_hourly['residual'] = df_hourly['no2_value'] - y_pred
    
    hourly_stats = df_hourly.groupby('hour').agg({
        'no2_value': ['mean', 'std', 'count'],
        'prediction': ['mean', 'std'],
        'residual': ['mean', 'std']
    }).round(2)
    
    # Aplanar nombres de columnas
    hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Valores por hora
    hours = hourly_stats.index
    ax1.plot(hours, hourly_stats['no2_value_mean'], 'o-', label='Real', linewidth=2)
    ax1.plot(hours, hourly_stats['prediction_mean'], 's--', label='Predicción', linewidth=2)
    ax1.fill_between(hours, 
                    hourly_stats['no2_value_mean'] - hourly_stats['no2_value_std'],
                    hourly_stats['no2_value_mean'] + hourly_stats['no2_value_std'],
                    alpha=0.2, label='±1 Std Real')
    ax1.set_xlabel('Hora del Día')
    ax1.set_ylabel('NO₂ Promedio (µg/m³)')
    ax1.set_title('Patrón Horario')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    
    # Residuos por hora
    ax2.plot(hours, hourly_stats['residual_mean'], 'o-', color='green', linewidth=2)
    ax2.fill_between(hours,
                    hourly_stats['residual_mean'] - hourly_stats['residual_std'],
                    hourly_stats['residual_mean'] + hourly_stats['residual_std'],
                    alpha=0.2, color='green')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Hora del Día')
    ax2.set_ylabel('Residuo Promedio (µg/m³)')
    ax2.set_title('Errores por Hora')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Mostrar tabla de estadísticas horarias
    with st.expander("📋 Estadísticas Horarias Detalladas"):
        st.dataframe(hourly_stats, use_container_width=True)


# ==================== COMPARACIÓN ENTRE MODELOS ====================

def show_model_comparison(models_metrics: Dict[str, Dict], 
                         title: str = "Comparación de Modelos"):
    """
    Muestra comparación entre múltiples modelos.
    
    Args:
        models_metrics: Diccionario con métricas de cada modelo
        title: Título de la comparación
    """
    st.subheader(f"⚖️ {title}")
    
    if len(models_metrics) < 2:
        st.warning("Se necesitan al menos 2 modelos para comparar.")
        return
    
    # Crear DataFrame para comparación
    comparison_data = []
    for model_name, metrics in models_metrics.items():
        comparison_data.append({
            'Modelo': model_name,
            'RMSE': metrics.get('rmse', np.nan),
            'R²': metrics.get('r2', np.nan),
            'MAE': metrics.get('mae', np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Gráfico de barras comparativo
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE
    axes[0].bar(comparison_df['Modelo'], comparison_df['RMSE'], 
               color='lightcoral', alpha=0.7)
    axes[0].set_title('RMSE por Modelo')
    axes[0].set_ylabel('RMSE (µg/m³)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # R²
    axes[1].bar(comparison_df['Modelo'], comparison_df['R²'], 
               color='lightblue', alpha=0.7)
    axes[1].set_title('R² por Modelo')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # MAE
    axes[2].bar(comparison_df['Modelo'], comparison_df['MAE'], 
               color='lightgreen', alpha=0.7)
    axes[2].set_title('MAE por Modelo')
    axes[2].set_ylabel('MAE (µg/m³)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Tabla de comparación
    st.dataframe(
        comparison_df.style.format({
            'RMSE': '{:.2f}',
            'R²': '{:.3f}',
            'MAE': '{:.2f}'
        }).highlight_min(subset=['RMSE', 'MAE']).highlight_max(subset=['R²']),
        use_container_width=True
    )


# ==================== UTILIDADES AUXILIARES ====================

def create_figure_with_style(figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Crea una figura con estilo consistente.
    
    Args:
        figsize: Tamaño de la figura
        
    Returns:
        Tupla con (figura, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300):
    """
    Guarda una figura con configuración estándar.
    
    Args:
        fig: Figura de matplotlib
        filename: Nombre del archivo
        dpi: Resolución
    """
    try:
        import os
        from ..config import FILE_PATHS
        
        os.makedirs(FILE_PATHS['figures_dir'], exist_ok=True)
        filepath = os.path.join(FILE_PATHS['figures_dir'], filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        return filepath
    except Exception as e:
        st.warning(f"No se pudo guardar la figura: {e}")
        return None


def show_data_overview(df: pd.DataFrame, title: str = "Overview del Dataset"):
    """
    Muestra overview estadístico del dataset.
    
    Args:
        df: DataFrame a analizar
        title: Título del overview
    """
    st.subheader(f"📊 {title}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total registros", f"{len(df):,}")
    
    with col2:
        if 'id_no2' in df.columns:
            st.metric("Sensores NO₂", df['id_no2'].nunique())
        else:
            st.metric("Columnas", len(df.columns))
    
    with col3:
        if 'id_trafico' in df.columns:
            st.metric("Sensores tráfico", df['id_trafico'].nunique())
        else:
            st.metric("Valores únicos", df.nunique().sum())
    
    with col4:
        if 'fecha' in df.columns:
            periodo_días = (df['fecha'].max() - df['fecha'].min()).days
            st.metric("Período (días)", f"{periodo_días:,}")
        else:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memoria (MB)", f"{memory_mb:.1f}")


def show_feature_distribution(df: pd.DataFrame, features: List[str], 
                            max_features: int = 6):
    """
    Muestra distribución de características seleccionadas.
    
    Args:
        df: DataFrame con los datos
        features: Lista de características a mostrar
        max_features: Número máximo de características a mostrar
    """
    if not features:
        return
    
    st.subheader("📊 Distribución de Variables Seleccionadas")
    
    # Limitar número de características
    display_features = features[:max_features]
    
    # Crear subplots
    n_features = len(display_features)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes if n_features > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(display_features):
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                axes[i].hist(df[feature].dropna(), bins=30, alpha=0.7, density=True)
                axes[i].set_title(f'Distribución: {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Densidad')
            else:
                value_counts = df[feature].value_counts().head(10)
                axes[i].bar(range(len(value_counts)), value_counts.values)
                axes[i].set_title(f'Top 10: {feature}')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45)
        else:
            axes[i].text(0.5, 0.5, f'Variable no disponible:\n{feature}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'No disponible: {feature}')
    
    # Ocultar subplots vacíos
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    if len(features) > max_features:
        st.info(f"Mostrando solo las primeras {max_features} de {len(features)} variables seleccionadas.") 