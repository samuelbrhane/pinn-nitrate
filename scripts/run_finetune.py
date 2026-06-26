import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from models.network import PINN
from utils.data_loader import DataHandler
from utils.losses import PINNLoss
from training.finetune import train_finetune


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
    
    # Load reaction weights
    model.load_state_dict(torch.load(f"{config.MODEL_DIR}/reaction_final.pt"))
    print(f"Loaded reaction weights")
    
    loss_fn = PINNLoss(model, config.DEVICE)
    model = train_finetune(model, train_loader, val_loader, loss_fn, epochs=config.EPOCHS_STAGE3)
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/pinn_final.pt")
    print(f"Fine-tuning complete! Final model saved!")