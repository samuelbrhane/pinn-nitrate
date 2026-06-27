import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from models.network import PINN
from utils.data_loader import DataHandler
from utils.losses import MultiSpeciesPINNLoss
from training.transport import train_transport


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
    
    print(f"Model created with INPUT_DIM={config.INPUT_DIM}, OUTPUT_DIM={config.OUTPUT_DIM}")
    
    # train_transport creates loss_fn internally, don't pass it
    model = train_transport(model, train_loader, val_loader, epochs=config.EPOCHS_STAGE1)
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/transport_final.pt")
    print(f"Transport training complete!")