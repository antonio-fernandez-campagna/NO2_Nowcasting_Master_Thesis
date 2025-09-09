"""
Ejemplos de uso del módulo BNN Unificado.

Este script demuestra cómo usar el módulo bayesian_neural_network_unified.py
para entrenar modelos BNN tanto individuales como globales (multi-sensor).
"""

from bayesian_neural_network_unified import BNNUnifiedTrainer, main_individual, main_global

def ejemplo_individual():
    """Ejemplo de entrenamiento individual para un sensor específico."""
    print("=" * 60)
    print("🔬 EJEMPLO: ENTRENAMIENTO BNN INDIVIDUAL")
    print("=" * 60)
    
    config = {
        'sensor_id': '28079008',
        'split_date': '2024-01-01',
        'outlier_method': 'iqr',
        'features': ['all'],  # Usar todas las características disponibles
        'K_predict': 100,
        'train_config': {
            'learning_rate': 0.01,
            'n_epochs': 50,
            'batch_size': 256,
            'hidden_dims': [64, 32],
            'activation': 'GELU',
            'beta': 1.0,
        }
    }
    
    # Ejecutar entrenamiento individual
    return main_individual(config)


def ejemplo_global():
    """Ejemplo de entrenamiento global para múltiples sensores."""
    print("=" * 60)
    print("🌍 EJEMPLO: ENTRENAMIENTO BNN GLOBAL MULTI-SENSOR")
    print("=" * 60)
    
    config = {
        'sensors_train': ['28079004', '28079008', '28079011'],
        'sensors_test': ['28079004', '28079008', '28079011'],
        'split_date': '2024-01-01',
        'outlier_method': 'iqr',
        'features': ['all'],  # Usar todas las características disponibles
        'K_predict': 100,
        'train_config': {
            'learning_rate': 0.01,
            'n_epochs': 50,
            'batch_size': 512,  # Mayor batch size para datos de múltiples sensores
            'hidden_dims': [128, 64, 32],  # Red más profunda para capturar patrones globales
            'activation': 'GELU',
            'beta': 1.0,
        }
    }
    
    # Ejecutar entrenamiento global
    return main_global(config)


def ejemplo_comparacion():
    """Ejemplo de comparación entre modelos individuales y globales."""
    print("=" * 60)
    print("⚖️ EJEMPLO: COMPARACIÓN INDIVIDUAL vs GLOBAL")
    print("=" * 60)
    
    # Entrenar modelo individual
    print("\n1️⃣ Entrenando modelo individual...")
    individual_results = ejemplo_individual()
    
    # Entrenar modelo global
    print("\n2️⃣ Entrenando modelo global...")
    global_results = ejemplo_global()
    
    print("\n📊 RESUMEN DE COMPARACIÓN:")
    print("-" * 40)
    
    if individual_results and global_results:
        ind_metrics = individual_results[2]  # metrics
        glob_metrics = global_results[2]     # global_metrics
        
        print(f"INDIVIDUAL - RMSE: {ind_metrics['rmse']:.2f}, R²: {ind_metrics['r2']:.3f}")
        print(f"GLOBAL     - RMSE: {glob_metrics['rmse']:.2f}, R²: {glob_metrics['r2']:.3f}")


def ejemplo_uso_avanzado():
    """Ejemplo de uso avanzado con configuraciones personalizadas."""
    print("=" * 60)
    print("🚀 EJEMPLO: USO AVANZADO - CONFIGURACIONES PERSONALIZADAS")
    print("=" * 60)
    
    # Inicializar trainer
    trainer = BNNUnifiedTrainer()
    
    # Cargar datos y obtener sensores disponibles
    trainer.df_master = trainer.load_data()
    sensores_disponibles = trainer.get_available_sensors()
    print(f"📍 Sensores disponibles: {sensores_disponibles}")
    
    # Configuración con características específicas
    features_meteorologicas = [
        't2m', 'd2m', 'sp', 'u10', 'v10', 'wind_speed', 'wind_direction',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekend'
    ]
    
    config_avanzado = {
        'sensor_id': '28079004',
        'split_date': '2024-01-01',
        'outlier_method': 'zscore',  # Método diferente de outliers
        'features': features_meteorologicas,  # Solo características meteorológicas y temporales
        'K_predict': 200,  # Más muestras Monte Carlo para mejor estimación de incertidumbre
        'train_config': {
            'learning_rate': 0.005,  # Learning rate más conservador
            'n_epochs': 100,         # Más épocas
            'batch_size': 128,       # Batch size más pequeño
            'hidden_dims': [256, 128, 64, 32],  # Red más profunda
            'activation': 'ReLU',    # Función de activación diferente
            'beta': 0.5,            # Peso menor para KL divergence
        }
    }
    
    print(f"🔧 Configuración avanzada:")
    print(f"   - Características: {len(features_meteorologicas)} variables específicas")
    print(f"   - Outliers: {config_avanzado['outlier_method']}")
    print(f"   - Épocas: {config_avanzado['train_config']['n_epochs']}")
    print(f"   - Arquitectura: {config_avanzado['train_config']['hidden_dims']}")
    print(f"   - Beta (KL weight): {config_avanzado['train_config']['beta']}")
    
    return main_individual(config_avanzado)


def ejemplo_predicciones_incertidumbre():
    """Ejemplo enfocado en análisis de incertidumbre."""
    print("=" * 60)
    print("🎯 EJEMPLO: ANÁLISIS DE INCERTIDUMBRE")
    print("=" * 60)
    
    # Configuración optimizada para capturar incertidumbre
    config_incertidumbre = {
        'sensor_id': '28079011',
        'split_date': '2024-01-01',
        'outlier_method': 'iqr',
        'features': ['all'],
        'K_predict': 500,  # Muchas más muestras para mejor estimación de incertidumbre
        'train_config': {
            'learning_rate': 0.01,
            'n_epochs': 75,
            'batch_size': 256,
            'hidden_dims': [100, 50, 25],
            'activation': 'GELU',
            'beta': 2.0,  # Mayor peso a la divergencia KL para mejor regularización Bayesiana
        }
    }
    
    print("🎲 Configuración optimizada para análisis de incertidumbre:")
    print(f"   - K_predict: {config_incertidumbre['K_predict']} (más muestras MC)")
    print(f"   - Beta: {config_incertidumbre['train_config']['beta']} (mayor regularización)")
    
    results = main_individual(config_incertidumbre)
    
    if results:
        trainer, model, metrics = results
        print(f"\n🔬 Análisis de incertidumbre completado:")
        print(f"   - Predicciones guardadas con incertidumbre epistémica")
        print(f"   - Archivo: ../predictions/bnn_predictions_with_epistemic_uncertainty_{config_incertidumbre['sensor_id']}.csv")
    
    return results


if __name__ == "__main__":
    print("🧠 EJEMPLOS DE USO - BNN UNIFICADO")
    print("=" * 60)
    print("Selecciona el ejemplo a ejecutar:")
    print("1. Entrenamiento Individual")
    print("2. Entrenamiento Global (Multi-sensor)")
    print("3. Comparación Individual vs Global")
    print("4. Uso Avanzado con Configuraciones Personalizadas")
    print("5. Análisis de Incertidumbre")
    print("0. Salir")
    
    try:
        opcion = input("\nIngresa tu opción (0-5): ").strip()
        
        if opcion == '1':
            ejemplo_individual()
        elif opcion == '2':
            ejemplo_global()
        elif opcion == '3':
            ejemplo_comparacion()
        elif opcion == '4':
            ejemplo_uso_avanzado()
        elif opcion == '5':
            ejemplo_predicciones_incertidumbre()
        elif opcion == '0':
            print("👋 ¡Hasta luego!")
        else:
            print("❌ Opción no válida. Ejecutando ejemplo individual por defecto.")
            ejemplo_individual()
            
    except KeyboardInterrupt:
        print("\n👋 Proceso interrumpido por el usuario.")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
