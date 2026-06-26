import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import time

def calculate_metrics(y_true, y_pred, y_true_denorm=None, y_pred_denorm=None):
    """Calculate all metrics: R², RMSE, Rel Error, etc"""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    
    # Normalized metrics
    r2 = r2_score(y_true_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    rel_error = np.mean(np.abs(y_true_np - y_pred_np) / (np.abs(y_true_np) + 1e-6))
    
    # Per-species metrics (if denormalized)
    species_metrics = {}
    if y_true_denorm is not None and y_pred_denorm is not None:
        y_true_denorm_np = y_true_denorm.cpu().numpy() if torch.is_tensor(y_true_denorm) else y_true_denorm
        y_pred_denorm_np = y_pred_denorm.cpu().numpy() if torch.is_tensor(y_pred_denorm) else y_pred_denorm
        
        species = ['NO3', 'DOC', 'Fe2+', 'N2']
        for i, sp in enumerate(species):
            species_metrics[sp] = {
                'R2': r2_score(y_true_denorm_np[:, i], y_pred_denorm_np[:, i]),
                'RMSE': np.sqrt(mean_squared_error(y_true_denorm_np[:, i], y_pred_denorm_np[:, i])),
                'RelError': np.mean(np.abs(y_true_denorm_np[:, i] - y_pred_denorm_np[:, i]) / (np.abs(y_true_denorm_np[:, i]) + 1e-6))
            }
    
    return {
        'Overall': {
            'R2': r2,
            'RMSE': rmse,
            'Rel_Error': rel_error
        },
        'PerSpecies': species_metrics
    }

def compare_methods(st_pinn_metrics, baseline_metrics):
    """Compare ST-PINN vs Baseline"""
    improvement = {
        'R2_improvement': ((st_pinn_metrics['Overall']['R2'] - baseline_metrics['Overall']['R2']) / baseline_metrics['Overall']['R2']) * 100,
        'RMSE_reduction': ((baseline_metrics['Overall']['RMSE'] - st_pinn_metrics['Overall']['RMSE']) / baseline_metrics['Overall']['RMSE']) * 100,
        'Error_reduction': ((baseline_metrics['Overall']['Rel_Error'] - st_pinn_metrics['Overall']['Rel_Error']) / baseline_metrics['Overall']['Rel_Error']) * 100
    }
    return improvement

def log_training_time(stage, time_seconds):
    """Log training time"""
    hours = time_seconds / 3600
    return f"Stage {stage}: {hours:.2f} hours ({time_seconds:.0f} sec)"