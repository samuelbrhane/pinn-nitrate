import torch
import torch.optim as optim
import config
from models.network import PINN
from utils.data_loader import DataHandler
from utils.losses import PINNLoss
from training.validate import evaluate
import os

def train_baseline(model, train_loader, val_loader, loss_fn, epochs=50000):
    """Baseline: Standard PINN (all at once, no DLW)"""
    print(f"\n{'='*60}")
    print(f"BASELINE PINN: Standard Training (50k epochs)")
    print(f"{'='*60}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(epochs):
        train_loss_total = 0
        for X_batch, Y_batch in train_loader:
            y_pred = model(X_batch)
            loss, _ = loss_fn(y_pred, Y_batch, epoch, is_stage1=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
        
        if epoch % 100 == 0:
            val_loss_total = 0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    y_pred_val = model(X_val)
                    loss_val, _ = loss_fn(y_pred_val, Y_val, epoch, is_stage1=True)
                    val_loss_total += loss_val.item()
            
            avg_train_loss = train_loss_total / len(train_loader)
            avg_val_loss = val_loss_total / len(val_loader)
            
            print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if epoch % config.CHECKPOINT_FREQ == 0 and epoch > 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.MODEL_DIR}/baseline_epoch{epoch}.pt")
    
    print(f"\nBaseline training complete!")
    return model

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
    
    print(f"Model created")
    
    loss_fn = PINNLoss()
    
    model = train_baseline(model, train_loader, val_loader, loss_fn, epochs=50000)
    
    evaluate(model, test_loader, loss_fn)
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/baseline_final.pt")
    print(f"Baseline saved!")