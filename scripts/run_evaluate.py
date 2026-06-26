import torch
import config
from models.network import PINN
from utils.data_loader import DataHandler
from utils.losses import PINNLoss
from utils.metrics import calculate_metrics
import json
import numpy as np
import os

def full_evaluation(model, test_loader, scaler_Y, method_name="ST-PINN"):
    """Full evaluation with denormalization"""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {method_name}")
    print(f"{'='*60}\n")
    
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            y_pred = model(X_test)
            all_y_true.append(Y_test.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
    
    y_true = np.vstack(all_y_true)
    y_pred = np.vstack(all_y_pred)
    
    # Denormalize
    y_true_denorm = scaler_Y.inverse_transform(y_true)
    y_pred_denorm = scaler_Y.inverse_transform(y_pred)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_true_denorm, y_pred_denorm)
    
    print(f"\nOverall Metrics:")
    print(f"  R²: {metrics['Overall']['R2']:.6f}")
    print(f"  RMSE: {metrics['Overall']['RMSE']:.6f}")
    print(f"  Relative Error: {metrics['Overall']['Rel_Error']:.6f}")
    
    if metrics['PerSpecies']:
        print(f"\nPer-Species Metrics:")
        for species, m in metrics['PerSpecies'].items():
            print(f"  {species}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/{method_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {config.RESULTS_DIR}/")
    
    return metrics

if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    print(f"Benchmark: {config.BENCHMARK}")
    
    dh = DataHandler(config.BENCHMARK)
    train_loader, val_loader, test_loader, scaler_X, scaler_Y = dh.create_dataloaders()
    
    model = PINN(
        input_dim=config.INPUT_DIM,
        output_dim=config.OUTPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(f"{config.MODEL_DIR}/pinn_final.pt"))
    print(f"Loaded ST-PINN final model")
    
    loss_fn = PINNLoss(model, config.DEVICE)
    
    # Evaluate
    metrics = full_evaluation(model, test_loader, scaler_Y, method_name="ST-PINN")