import torch
import config
from models.network import PINN
from utils.data_loader import DataHandler
from utils.losses import PINNLoss
from training.train_stage1 import train_stage1
from training.validate import evaluate
import os

if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    print(f"Benchmark: {config.BENCHMARK}")
    
    # Load data
    dh = DataHandler(config.BENCHMARK)
    train_loader, val_loader, test_loader, scaler_X, scaler_Y = dh.create_dataloaders()
    
    # Create model
    model = PINN(
        input_dim=config.INPUT_DIM,
        output_dim=config.OUTPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    ).to(config.DEVICE)
    
    print(f"✓ Model created")
    
    # Loss function
    loss_fn = PINNLoss()
    
    # Train Stage 1
    model = train_stage1(model, train_loader, val_loader, loss_fn, epochs=config.EPOCHS_STAGE1)
    
    # Evaluate
    evaluate(model, test_loader, loss_fn)
    
    # Save
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/stage1_final.pt")
    print(f"✓ Model saved!")