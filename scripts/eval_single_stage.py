import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from models.network import PINN
from utils.data_loader import DataHandler
from utils.metrics import calculate_metrics
import json
import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj

def evaluate_stage(model, test_loader, scaler_Y, stage_name):
    """Evaluate model on test set"""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {stage_name.upper()}")
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
    
    y_true_denorm = scaler_Y.inverse_transform(y_true)
    y_pred_denorm = scaler_Y.inverse_transform(y_pred)
    
    metrics = calculate_metrics(y_true, y_pred, y_true_denorm, y_pred_denorm)
    
    print(f"Overall Metrics:")
    print(f"  R²: {metrics['Overall']['R2']:.6f}")
    print(f"  RMSE: {metrics['Overall']['RMSE']:.6f}")
    print(f"  Relative Error: {metrics['Overall']['Rel_Error']:.6f}")
    
    if metrics['PerSpecies']:
        print(f"\nPer-Species Metrics:")
        for species, m in metrics['PerSpecies'].items():
            print(f"  {species}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    metrics_serializable = convert_to_serializable(metrics)
    
    with open(f"{config.RESULTS_DIR}/{stage_name}_metrics.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\nMetrics saved to {config.RESULTS_DIR}/{stage_name}_metrics.json")
    
    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_single_stage.py [stage]")
        print("Stages: transport, reaction, finetune, baseline")
        sys.exit(1)
    
    stage = sys.argv[1].lower()
    valid_stages = ['transport', 'reaction', 'finetune', 'baseline']
    
    if stage not in valid_stages:
        print(f"Invalid stage: {stage}")
        print(f"Valid stages: {', '.join(valid_stages)}")
        sys.exit(1)
    
    model_file = f"{config.MODEL_DIR}/{stage}_final.pt"
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        sys.exit(1)
    
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
    
    model.load_state_dict(torch.load(model_file))
    print(f"Loaded {stage} model\n")
    
    metrics = evaluate_stage(model, test_loader, scaler_Y, stage)