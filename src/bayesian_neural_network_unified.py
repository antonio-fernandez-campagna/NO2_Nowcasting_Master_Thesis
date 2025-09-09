"""
Módulo unificado para entrenamiento de Redes Neuronales Bayesianas (BNN).

Este módulo proporciona una interfaz para entrenar y analizar modelos BNN
tanto individuales (por sensor) como globales (multi-sensor) para predecir 
niveles de NO2, capturando tanto la predicción como la incertidumbre mediante
Inferencia Variacional de Campo Medio (MFVI).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta
import seaborn as sns
import joblib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Union
import warnings
import statsmodels.api as sm
from scipy.stats import zscore
import math
import argparse

warnings.filterwarnings('ignore')
torch.classes.__path__ = [] # add this line to manually set it to empty.

# Constante para estabilidad numérica
EPS = 1e-6

# ==================== COMPONENTES BNN (Basado en bayesian_example.py) ====================

class MFVILinear(nn.Module):
    """Capa lineal Bayesiana usando Inferencia Variacional de Campo Medio (MFVI)."""

    def __init__(self, dim_in, dim_out, prior_weight_std=1.0, prior_bias_std=1.0, init_std=0.05, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.weight_mean = nn.Parameter(torch.empty((dim_out, dim_in), **factory_kwargs))
        self.bias_mean = nn.Parameter(torch.empty(dim_out, **factory_kwargs))
        self._weight_std_param = nn.Parameter(torch.empty((dim_out, dim_in), **factory_kwargs))
        self._bias_std_param = nn.Parameter(torch.empty(dim_out, **factory_kwargs))
        
        self.reset_parameters(init_std)

        prior_mean = 0.0
        self.register_buffer('prior_weight_mean', torch.full_like(self.weight_mean, prior_mean))
        self.register_buffer('prior_weight_std', torch.full_like(self._weight_std_param, prior_weight_std))
        self.register_buffer('prior_bias_mean', torch.full_like(self.bias_mean, prior_mean))
        self.register_buffer('prior_bias_std', torch.full_like(self._bias_std_param, prior_bias_std))

    def reset_parameters(self, init_std=0.05):
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        bound = self.dim_in ** -0.5 if self.dim_in > 0 else 0
        nn.init.uniform_(self.bias_mean, -bound, bound)
        _init_std_param = np.log(init_std)
        self._weight_std_param.data = torch.full_like(self.weight_mean, _init_std_param)
        self._bias_std_param.data = torch.full_like(self.bias_mean, _init_std_param)

    @property
    def weight_std(self):
        return torch.clamp(torch.exp(self._weight_std_param), min=EPS)

    @property
    def bias_std(self):
        return torch.clamp(torch.exp(self._bias_std_param), min=EPS)

    def kl_divergence(self):
        q_weight = dist.Normal(self.weight_mean, self.weight_std)
        p_weight = dist.Normal(self.prior_weight_mean, self.prior_weight_std)
        kl = dist.kl_divergence(q_weight, p_weight).sum()
        
        q_bias = dist.Normal(self.bias_mean, self.bias_std)
        p_bias = dist.Normal(self.prior_bias_mean, self.prior_bias_std)
        kl += dist.kl_divergence(q_bias, p_bias).sum()
        return kl

    def forward(self, input):
        weight = self._normal_sample(self.weight_mean, self.weight_std)
        bias = self._normal_sample(self.bias_mean, self.bias_std)
        return F.linear(input, weight, bias)

    def _normal_sample(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

def make_mfvi_bnn(layer_sizes, activation='GELU', **layer_kwargs):
    nonlinearity = getattr(nn, activation)() if isinstance(activation, str) else activation
    net = nn.Sequential()
    for i, (dim_in, dim_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        net.add_module(f'MFVILinear{i}', MFVILinear(dim_in, dim_out, **layer_kwargs))
        if i < len(layer_sizes) - 2:
            net.add_module(f'Nonlinarity{i}', nonlinearity)
    return net

def kl_divergence_model(bnn):
    kl = 0.0
    for module in bnn.modules():
        if hasattr(module, 'kl_divergence'):
            kl += module.kl_divergence()
    return kl

def gauss_loglik(y, y_pred, log_noise_var):
    l2_dist = (y - y_pred).pow(2).sum(-1)
    return -0.5 * (log_noise_var + math.log(2 * math.pi) + l2_dist * torch.exp(-log_noise_var))

def test_nll(y, y_pred, log_noise_var):
    nll_samples = -gauss_loglik(y, y_pred, log_noise_var) # Shape (K, N_test)
    nll = -torch.logsumexp(-nll_samples, dim=0) + math.log(nll_samples.shape[0])
    return nll.mean()

# ==================== CLASE DATASET PYTORCH ====================

class BayesianDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# ==================== CLASE BNN TRAINER UNIFICADO ====================

class BNNUnifiedTrainer:
    """
    Clase unificada para entrenamiento de BNNs que maneja tanto modelos individuales 
    como globales (multi-sensor) con una interfaz consistente.
    """
    
    def __init__(self):
        self.df_master = None
        self.model = None
        self.scaler_dict = {}
        self.scaler_target = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def load_data(self) -> pd.DataFrame:
        """Cargar datos desde el archivo parquet."""
        try:
            df = pd.read_parquet('../data/super_processed/7_4_no2_with_traffic_and_1meteo_and_1trafic_id.parquet')
            df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        except Exception as e:
            print(f"Error al cargar los datos: {str(e)}")
            raise e
    
    def get_available_sensors(self) -> List[str]:
        """Obtener lista de sensores disponibles en los datos."""
        if self.df_master is None:
            self.df_master = self.load_data()
        return sorted(self.df_master['id_no2'].unique().tolist())
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear características temporales cíclicas."""
        df = df.copy()
        df['day_of_week'] = df['fecha'].dt.dayofweek
        df['day_of_year'] = df['fecha'].dt.dayofyear
        df['month'] = df['fecha'].dt.month
        df['hour'] = df['fecha'].dt.hour
        df['day'] = df['fecha'].dt.day
        df['weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3) # 1:winter, 2:spring, 3:summer, 4:autumn
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df

    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertir unidades de las variables meteorológicas."""
        df = df.copy()
        for col in ['d2m', 't2m']:
            if col in df.columns: df[col] -= 273.15
        for col in ['ssr', 'ssrd']:
            if col in df.columns: df[col] /= 3600
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2) * 3.6
            df['wind_direction'] = (270 - np.arctan2(df['v10'], df['u10']) * 180 / np.pi) % 360
        if 'sp' in df.columns: df['sp'] /= 100
        if 'tp' in df.columns: df['tp'] *= 1000
        return df

    def remove_outliers(self, df: pd.DataFrame, method: str, columns: List[str] = None) -> pd.DataFrame:
        """Remover outliers usando el método especificado."""
        if method == 'none': 
            return df
        
        if columns is None:
            columns = ['no2_value']
            
        df_filtered = df.copy()
        for col in columns:
            if col not in df_filtered.columns: 
                continue
            if method == 'iqr':
                Q1, Q3 = df_filtered[col].quantile(0.25), df_filtered[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]
            elif method == 'zscore':
                df_filtered = df_filtered[np.abs(zscore(df_filtered[col], nan_policy='omit')) < 3]
        return df_filtered

    def split_data(self, df: pd.DataFrame, split_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Dividir datos por fecha."""
        train = df[df['fecha'] < split_date].copy()
        test = df[df['fecha'] >= split_date].copy()
        return train, test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Escalar características usando StandardScaler."""
        scaler_dict = {}
        X_train_s, X_test_s = X_train.copy(), X_test.copy()
        for feature in features:
            if feature in X_train.columns and pd.api.types.is_numeric_dtype(X_train[feature]):
                scaler = StandardScaler()
                X_train_s[feature] = scaler.fit_transform(X_train[[feature]]).flatten()
                X_test_s[feature] = scaler.transform(X_test[[feature]]).flatten()
                scaler_dict[feature] = scaler
        return X_train_s, X_test_s, scaler_dict

    def scale_target(self, y_train: pd.Series) -> Tuple[pd.Series, StandardScaler]:
        """Escalar variable objetivo."""
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        return pd.Series(y_scaled, index=y_train.index, name=y_train.name), scaler

    def prepare_individual_data(self, config: Dict, selected_features: List[str]) -> Dict:
        """Preparar datos para entrenamiento individual (un sensor)."""
        print(f"\n📊 Preparando datos para sensor individual: {config['sensor_id']}")
        
        # Filtrar por sensor
        df_sensor = self.df_master[self.df_master['id_no2'] == config['sensor_id']].copy()
        if df_sensor.empty:
            print(f"❌ Error: No se encontraron datos para el sensor {config['sensor_id']}")
            return {}

        print(f"📊 Datos originales del sensor: {len(df_sensor)}")
        
        # Aplicar preprocesamiento
        df_processed = self.convert_units(df_sensor)
        df_processed = self.create_cyclical_features(df_processed)
        
        # División temporal
        train_df, test_df = self.split_data(df_processed, pd.to_datetime(config['split_date']))
        
        # Remover outliers solo en entrenamiento
        outliers_removed = 0
        if config['outlier_method'] != 'none':
            len_before = len(train_df)
            train_df = self.remove_outliers(train_df, config['outlier_method'], ['no2_value'])
            outliers_removed = len_before - len(train_df)
            print(f"❌ Outliers eliminados: {outliers_removed}")
        
        print(f"📅 Datos entrenamiento finales: {len(train_df)}")
        print(f"📅 Datos test finales: {len(test_df)}")
        
        return {
            'train_df': train_df,
            'test_df': test_df,
            'outliers_removed': outliers_removed,
            'mode': 'individual',
            'sensor_id': config['sensor_id']
        }

    def prepare_global_data(self, config: Dict, selected_features: List[str]) -> Dict:
        """Preparar datos para entrenamiento global (múltiples sensores)."""
        print(f"\n🌍 Preparando datos para modelo global multi-sensor")
        print(f"   Sensores entrenamiento: {config['sensors_train']}")
        print(f"   Sensores test: {config['sensors_test']}")
        
        # Preparar datos de entrenamiento y test
        df_train = self.df_master[self.df_master['id_no2'].isin(config['sensors_train'])].copy()
        df_test = self.df_master[self.df_master['id_no2'].isin(config['sensors_test'])].copy()
        
        print(f"📊 Datos entrenamiento originales: {len(df_train)}")
        print(f"📊 Datos test originales: {len(df_test)}")
        
        # Aplicar preprocesamiento
        df_train = self.convert_units(df_train)
        df_train = self.create_cyclical_features(df_train)
        df_test = self.convert_units(df_test)
        df_test = self.create_cyclical_features(df_test)
        
        # División temporal adicional
        df_train = df_train[df_train['fecha'] < pd.to_datetime(config['split_date'])].copy()
        df_test = df_test[df_test['fecha'] >= pd.to_datetime(config['split_date'])].copy()
        
        # Validar que todas las features existen
        missing_features_train = [f for f in selected_features if f not in df_train.columns]
        missing_features_test = [f for f in selected_features if f not in df_test.columns]
        
        if missing_features_train or missing_features_test:
            print(f"❌ Features faltantes en train: {missing_features_train}")
            print(f"❌ Features faltantes en test: {missing_features_test}")
            return {}
        
        # Remover outliers solo en entrenamiento
        outliers_removed = 0
        if config['outlier_method'] != 'none':
            len_before = len(df_train)
            df_train = self.remove_outliers(df_train, config['outlier_method'], ['no2_value'])
            outliers_removed = len_before - len(df_train)
            print(f"❌ Outliers eliminados: {outliers_removed}")
        
        print(f"📅 Datos entrenamiento finales: {len(df_train)}")
        print(f"📅 Datos test finales: {len(df_test)}")
        
        return {
            'train_df': df_train,
            'test_df': df_test,
            'outliers_removed': outliers_removed,
            'mode': 'global',
            'sensors_train': config['sensors_train'],
            'sensors_test': config['sensors_test']
        }

    def train_bnn_model(self, X_train, y_train, train_config):
        """Entrenar el modelo BNN."""
        # Setup
        dataset = BayesianDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)
        
        layer_sizes = [X_train.shape[1]] + train_config['hidden_dims'] + [1]
        model = make_mfvi_bnn(layer_sizes, activation=train_config['activation'], device=self.device).to(self.device)
        log_noise_var = nn.Parameter(torch.ones(1, device=self.device) * -3.0)
        
        params = list(model.parameters()) + [log_noise_var]
        optimizer = torch.optim.Adam(params, lr=train_config['learning_rate'])
        
        # Training Loop
        logs = []
        N_data = len(dataset)
        
        print("🚀 Iniciando entrenamiento BNN...")
        for epoch in range(train_config['n_epochs']):
            model.train()
            epoch_nll, epoch_kl = 0, 0
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                
                y_pred = model(x_batch)
                nll = -gauss_loglik(y_batch, y_pred, log_noise_var).mean()
                kl = kl_divergence_model(model)
                loss = nll + train_config['beta'] * kl / N_data
                
                loss.backward()
                optimizer.step()
                
                epoch_nll += nll.item() * len(x_batch)
                epoch_kl += kl.item()
            
            logs.append({'nll': epoch_nll / N_data, 'kl': epoch_kl / N_data})
            if (epoch + 1) % 50 == 0:
                 print(f"   Epoch {epoch+1}/{train_config['n_epochs']} | NLL: {logs[-1]['nll']:.3f} | KL: {logs[-1]['kl']:.3f}")
        
        print("✅ Entrenamiento completado.")
        return model, log_noise_var, logs

    def predict(self, model, X_test, K, log_noise_var):
        """Realizar predicciones con el modelo BNN."""
        model.eval()
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            y_preds = torch.stack([model(X_test_tensor) for _ in range(K)], dim=0)
        
        pred_mean = y_preds.mean(0)
        epistemic_uncertainty = y_preds.var(0).sqrt()
        aleatoric_uncertainty = torch.exp(0.5 * log_noise_var).expand_as(pred_mean)
        total_uncertainty = (epistemic_uncertainty**2 + aleatoric_uncertainty**2).sqrt()
        
        return {
            'y_preds_all': y_preds,
            'mean': pred_mean,
            'epistemic_std': epistemic_uncertainty,
            'aleatoric_std': aleatoric_uncertainty,
            'total_std': total_uncertainty
        }

    def evaluate_model(self, predictions, y_test, y_test_scaled, scaler_target, log_noise_var, sensor_id: str = None):
        """Evaluar el modelo y calcular métricas."""
        # Unscale predictions
        pred_mean_scaled = predictions['mean'].detach().cpu().numpy()
        pred_mean = scaler_target.inverse_transform(pred_mean_scaled).flatten()
        
        total_std_scaled = predictions['total_std'].detach().cpu().numpy().flatten()
        unscaled_std = total_std_scaled * scaler_target.scale_[0]
        
        epistemic_std = predictions['epistemic_std'].detach().cpu().numpy().flatten()
        epistemic_unescaled_std = epistemic_std * scaler_target.scale_[0]

        # Create DataFrame for saving predictions
        df_preds = pd.DataFrame({
            'prediction': pred_mean,
            'epistemic_uncertainty': epistemic_unescaled_std
        })

        # Save predictions
        sensor_suffix = f"_{sensor_id}" if sensor_id else "_global"
        filename = f'../predictions/bnn_predictions_with_epistemic_uncertainty{sensor_suffix}.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_preds.to_csv(filename, index=False)
        print(f"💾 Predicciones guardadas en: {filename}")

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, pred_mean))
        r2 = r2_score(y_test, pred_mean)
        mae = mean_absolute_error(y_test, pred_mean)
        
        # Test NLL
        y_test_tensor = torch.tensor(y_test_scaled.values, dtype=torch.float32).reshape(-1, 1).to(self.device)
        nll = test_nll(y_test_tensor, predictions['y_preds_all'], log_noise_var).item()
        
        return {
            'rmse': rmse, 'r2': r2, 'mae': mae, 'test_nll': nll,
            'y_pred': pred_mean, 'y_pred_std': unscaled_std,
            'predictions_df': df_preds
        }

    def evaluate_global_by_sensor(self, data_prep: Dict, model, log_noise_var, scaler_dict, 
                                scaler_target, selected_features, K_predict: int) -> Dict:
        """Evaluar modelo global por sensor individual."""
        results_by_sensor = {}
        test_df = data_prep['test_df']
        
        print(f"\n🔍 Evaluando modelo global por sensor individual...")
        
        for sensor_id in data_prep['sensors_test']:
            print(f"   📊 Evaluando sensor: {sensor_id}")
            
            # Filtrar datos del sensor específico
            sensor_test_df = test_df[test_df['id_no2'] == sensor_id].copy()
            
            if len(sensor_test_df) == 0:
                print(f"      ⚠️ No hay datos de test para sensor {sensor_id}")
                continue
            
            # Preparar features para este sensor
            X_test_sensor = sensor_test_df[selected_features].copy()
            y_test_sensor = sensor_test_df['no2_value'].copy()
            
            # Aplicar escalado (usando los scalers del entrenamiento global)
            X_test_sensor_scaled = X_test_sensor.copy()
            for feature in selected_features:
                if feature in scaler_dict:
                    X_test_sensor_scaled[feature] = scaler_dict[feature].transform(X_test_sensor[[feature]]).flatten()
            
            y_test_sensor_scaled = pd.Series(
                scaler_target.transform(y_test_sensor.values.reshape(-1, 1)).flatten(),
                index=y_test_sensor.index,
                name=y_test_sensor.name
            )
            
            # Realizar predicciones
            predictions = self.predict(model, X_test_sensor_scaled, K_predict, log_noise_var)
            
            # Evaluar métricas para este sensor
            metrics = self.evaluate_model(predictions, y_test_sensor, y_test_sensor_scaled, 
                                        scaler_target, log_noise_var, sensor_id)
            
            results_by_sensor[sensor_id] = {
                'metrics': metrics,
                'test_df': sensor_test_df,
                'n_samples': len(sensor_test_df)
            }
            
            print(f"      RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.2f}")
        
        return results_by_sensor
    
    def save_model(self, path: str, model, log_noise_var, scaler_dict, scaler_target, 
                   feature_names: List[str], model_config: Dict = None):
        """Guardar modelo entrenado."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_state = {
            'model_state_dict': model.state_dict(),
            'log_noise_var': log_noise_var,
            'scaler_dict': scaler_dict,
            'scaler_target': scaler_target,
            'feature_names': feature_names,
            'model_config': model_config or {}
        }
        joblib.dump(model_state, path)
        print(f"💾 Modelo guardado en: {path}")

    def load_model(self, path: str, layer_sizes: List[int], activation: str = 'GELU'):
        """Cargar modelo guardado."""
        if not os.path.exists(path):
            print(f"❌ Modelo no encontrado en: {path}")
            return None
        
        model_state = joblib.load(path)
        model = make_mfvi_bnn(layer_sizes, activation=activation, device=self.device).to(self.device)
        model.load_state_dict(model_state['model_state_dict'])
        log_noise_var = model_state['log_noise_var'].to(self.device)
        
        return (model, log_noise_var, model_state['scaler_dict'], 
                model_state['scaler_target'], model_state['feature_names'])

# ==================== FUNCIONES DE VISUALIZACIÓN Y REPORTE ====================

def print_model_metrics(metrics: Dict, title: str = "Métricas de Evaluación"):
    """Mostrar métricas del modelo."""
    print(f"\n📊 {title}")
    print(f"  - RMSE: {metrics['rmse']:.2f} µg/m³ (Error Cuadrático Medio)")
    print(f"  - R² Score: {metrics['r2']:.3f} (Coeficiente de Determinación)")
    print(f"  - MAE: {metrics['mae']:.2f} µg/m³ (Error Absoluto Medio)")
    print(f"  - Test NLL: {metrics['test_nll']:.3f} (Log-verosimilitud Negativa en Test)")

def print_global_summary(results_by_sensor: Dict):
    """Mostrar resumen de resultados globales por sensor."""
    print(f"\n🌍 Resumen Modelo Global - Resultados por Sensor")
    print("="*60)
    
    # Calcular métricas agregadas
    all_rmse = [results_by_sensor[sensor]['metrics']['rmse'] for sensor in results_by_sensor]
    all_r2 = [results_by_sensor[sensor]['metrics']['r2'] for sensor in results_by_sensor]
    all_mae = [results_by_sensor[sensor]['metrics']['mae'] for sensor in results_by_sensor]
    
    print(f"📊 Métricas Promedio:")
    print(f"  - RMSE Promedio: {np.mean(all_rmse):.2f} µg/m³ (±{np.std(all_rmse):.2f})")
    print(f"  - R² Promedio: {np.mean(all_r2):.3f} (±{np.std(all_r2):.3f})")
    print(f"  - MAE Promedio: {np.mean(all_mae):.2f} µg/m³ (±{np.std(all_mae):.2f})")
    
    print(f"\n📋 Detalle por Sensor:")
    for sensor_id in sorted(results_by_sensor.keys()):
        result = results_by_sensor[sensor_id]
        metrics = result['metrics']
        print(f"  {sensor_id}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}, "
              f"MAE={metrics['mae']:.2f}, n={result['n_samples']}")

def save_training_loss_plot(logs: List[Dict], filename: str):
    """Guardar gráfico de curvas de entrenamiento."""
    print(f"📉 Guardando curvas de entrenamiento en {filename}")
    log_df = pd.DataFrame(logs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(log_df['nll'])
    ax1.set_title("Negative Log-Likelihood (NLL)")
    ax1.set_xlabel("Época")
    ax2.plot(log_df['kl'])
    ax2.set_title("KL Divergence")
    ax2.set_xlabel("Época")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close(fig)

# ==================== FUNCIONES PRINCIPALES ====================

def main_individual(config: Dict):
    """Función principal para entrenamiento individual."""
    print("🧠 Entrenamiento BNN Individual")
    trainer = BNNUnifiedTrainer()
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    trainer.df_master = trainer.load_data()
    if trainer.df_master.empty: 
        return
    
    print(f"\n⚙️ Configuración Individual:")
    print(f"  - Sensor: {config['sensor_id']}")
    print(f"  - Fecha división: {config['split_date']}")
    print(f"  - Outliers: {config['outlier_method']}")
    
    # Definir features ANTES de preparar datos
    # Necesitamos una muestra de datos para obtener las columnas disponibles
    sample_data = trainer.df_master[trainer.df_master['id_no2'] == config['sensor_id']].head(100).copy()
    if sample_data.empty:
        print(f"❌ Error: No se encontraron datos para el sensor {config['sensor_id']}")
        return
    
    sample_data = trainer.convert_units(sample_data)
    sample_data = trainer.create_cyclical_features(sample_data)
    
    all_features = [c for c in sample_data.columns 
                   if c not in ['fecha', 'id_no2', 'no2_value'] 
                   and pd.api.types.is_numeric_dtype(sample_data[c])]
    
    if config['features'] == ['all']:
        selected_features = all_features
    else:
        selected_features = [f for f in config['features'] if f in all_features]
    
    print(f"\n🔧 Usando {len(selected_features)} características")
    
    # Preparar datos con las features ya definidas
    data_prep = trainer.prepare_individual_data(config, selected_features)
    if not data_prep:
        return
    
    # Preparar matrices
    X_train = data_prep['train_df'][selected_features]
    y_train = data_prep['train_df']['no2_value']
    X_test = data_prep['test_df'][selected_features]
    y_test = data_prep['test_df']['no2_value']
    
    # Escalar datos
    X_train_scaled, X_test_scaled, scaler_dict = trainer.scale_features(X_train, X_test, selected_features)
    y_train_scaled, scaler_target = trainer.scale_target(y_train)
    y_test_scaled = pd.Series(
        scaler_target.transform(y_test.values.reshape(-1, 1)).flatten(),
        index=y_test.index, name=y_test.name
    )
    
    print(f"📊 Datos: Entrenamiento={len(X_train_scaled)}, Test={len(X_test_scaled)}")
    
    # Entrenar modelo
    model, log_noise_var, logs = trainer.train_bnn_model(X_train_scaled, y_train_scaled, config['train_config'])
    
    # Evaluar modelo
    print("\n🔍 Evaluando modelo...")
    predictions = trainer.predict(model, X_test_scaled, config['K_predict'], log_noise_var)
    metrics = trainer.evaluate_model(predictions, y_test, y_test_scaled, scaler_target, log_noise_var, config['sensor_id'])
    
    # Mostrar resultados
    print_model_metrics(metrics)
    
    # Guardar modelo
    model_path = f"../models/bnn_model_{config['sensor_id']}.pkl"
    trainer.save_model(model_path, model, log_noise_var, scaler_dict, scaler_target, 
                      selected_features, config)
    
    # Guardar gráficos de entrenamiento
    save_training_loss_plot(logs, f"../models/bnn_training_loss_{config['sensor_id']}.png")
    
    print("\n✅ Proceso individual completado.")
    return trainer, model, metrics


def main_global(config: Dict):
    """Función principal para entrenamiento global (multi-sensor)."""
    print("🌍 Entrenamiento BNN Global Multi-Sensor")
    trainer = BNNUnifiedTrainer()
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    trainer.df_master = trainer.load_data()
    if trainer.df_master.empty: 
        return
    
    print(f"\n⚙️ Configuración Global:")
    print(f"  - Sensores entrenamiento: {config['sensors_train']}")
    print(f"  - Sensores test: {config['sensors_test']}")
    print(f"  - Fecha división: {config['split_date']}")
    print(f"  - Outliers: {config['outlier_method']}")
    
    # Definir features ANTES de preparar datos
    # Necesitamos una muestra de datos para obtener las columnas disponibles
    sample_data = trainer.df_master[trainer.df_master['id_no2'].isin(config['sensors_train'])].head(100).copy()
    sample_data = trainer.convert_units(sample_data)
    sample_data = trainer.create_cyclical_features(sample_data)
    
    all_features = [c for c in sample_data.columns 
                   if c not in ['fecha', 'id_no2', 'no2_value'] 
                   and pd.api.types.is_numeric_dtype(sample_data[c])]
    
    if config['features'] == ['all']:
        selected_features = all_features
    else:
        selected_features = [f for f in config['features'] if f in all_features]
    
    print(f"\n🔧 Usando {len(selected_features)} características")
    
    # Preparar datos con las features ya definidas
    data_prep = trainer.prepare_global_data(config, selected_features)
    if not data_prep:
        return
    
    # Preparar matrices
    X_train = data_prep['train_df'][selected_features]
    y_train = data_prep['train_df']['no2_value']
    X_test = data_prep['test_df'][selected_features]
    y_test = data_prep['test_df']['no2_value']
    
    # Escalar datos
    X_train_scaled, X_test_scaled, scaler_dict = trainer.scale_features(X_train, X_test, selected_features)
    y_train_scaled, scaler_target = trainer.scale_target(y_train)
    y_test_scaled = pd.Series(
        scaler_target.transform(y_test.values.reshape(-1, 1)).flatten(),
        index=y_test.index, name=y_test.name
    )
    
    print(f"📊 Datos: Entrenamiento={len(X_train_scaled)}, Test={len(X_test_scaled)}")
    
    # Entrenar modelo
    model, log_noise_var, logs = trainer.train_bnn_model(X_train_scaled, y_train_scaled, config['train_config'])
    
    # Evaluar modelo globalmente
    print("\n🔍 Evaluando modelo global...")
    predictions = trainer.predict(model, X_test_scaled, config['K_predict'], log_noise_var)
    global_metrics = trainer.evaluate_model(predictions, y_test, y_test_scaled, scaler_target, log_noise_var, "global")
    
    # Evaluar modelo por sensor individual
    sensor_results = trainer.evaluate_global_by_sensor(
        data_prep, model, log_noise_var, scaler_dict, scaler_target, 
        selected_features, config['K_predict']
    )
    
    # Mostrar resultados
    print_model_metrics(global_metrics, "Métricas Globales")
    print_global_summary(sensor_results)
    
    # Guardar modelo
    model_path = f"../models/bnn_model_global.pkl"
    trainer.save_model(model_path, model, log_noise_var, scaler_dict, scaler_target, 
                      selected_features, config)
    
    # Guardar gráficos de entrenamiento
    save_training_loss_plot(logs, f"../models/bnn_training_loss_global.png")
    
    print("\n✅ Proceso global completado.")
    return trainer, model, global_metrics, sensor_results



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
            'n_epochs': 30,
            'batch_size': 512,  # Mayor batch size para datos de múltiples sensores
            'hidden_dims': [128, 64, 32],  # Red más profunda para capturar patrones globales
            'activation': 'GELU',
            'beta': 1.0,
        }
    }
    
    # Ejecutar entrenamiento global
    return main_global(config)

ejemplo_global()