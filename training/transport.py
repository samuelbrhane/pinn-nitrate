import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import config
import json

def train_transport(model, train_loader, val_loader, loss_fn, epochs=20000):
    """Transport Training: Train movement/advection-dispersion component"""
    print(f"\n{'='*60}")
    print(f"Transport Training (20k epochs)")
    print(f"{'='*60}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    loss_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        train_loss_total = 0
        num_batches = 0
        for X_batch, Y_batch in train_loader:
            y_pred = model(X_batch)
            
            x_colloc = torch.rand(config.N_COLLOCATION, 3, device=config.DEVICE) * 100
            x_colloc[:, 2] = 0.5
            
            loss, _ = loss_fn(X_batch, y_pred, Y_batch, x_colloc, epoch, compute_pde=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss_total / num_batches
        
        if epoch % 100 == 0:
            val_loss_total = 0
            num_val_batches = 0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    y_pred_val = model(X_val)
                    x_colloc_dummy = torch.zeros(1, 3, device=config.DEVICE)
                    loss_val, _ = loss_fn(X_val, y_pred_val, Y_val, x_colloc_dummy, epoch, compute_pde=False)
                    val_loss_total += loss_val.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss_total / num_val_batches
            
            loss_history['epochs'].append(epoch)
            loss_history['train_loss'].append(avg_train_loss)
            loss_history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if epoch % config.CHECKPOINT_FREQ == 0 and epoch > 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.MODEL_DIR}/transport_epoch{epoch}.pt")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/transport_loss.json", 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"\nTransport training complete!")
    print(f"Loss history saved to {config.RESULTS_DIR}/transport_loss.json")
    
    return model