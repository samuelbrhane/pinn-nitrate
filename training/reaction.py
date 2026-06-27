import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import config
import json
import time

def train_reaction(model, train_loader, val_loader, loss_fn, epochs=5000):
    """Reaction Training: Transport layers with reduced LR, reaction layers with full LR"""
    print(f"\n{'='*60}")
    print(f"Reaction Training (5k epochs, PDE per epoch)")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Progressive freezing: Reduced learning rate for transport layers
    transport_params = [p for n, p in model.named_parameters() if 'network.0' in n or 'network.1' in n]
    reaction_params = [p for n, p in model.named_parameters() if 'network.0' not in n and 'network.1' not in n]
    
    optimizer = optim.Adam([
        {'params': reaction_params, 'lr': config.LEARNING_RATE},
        {'params': transport_params, 'lr': config.LEARNING_RATE * 0.1}
    ])
    
    start_epoch = 0
    loss_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # Check for checkpoint
    checkpoint_path = f"{config.MODEL_DIR}/reaction_latest.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss_history = checkpoint['loss_history']
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        # Generate collocation points ONCE per epoch
        x_colloc = torch.rand(config.N_COLLOCATION, 3, device=config.DEVICE) * 100
        x_colloc[:, 2] = 0.5
        
        train_loss_total = 0
        num_batches = 0
        for X_batch, Y_batch in train_loader:
            y_pred = model(X_batch)
            
            loss, _ = loss_fn(X_batch, y_pred, Y_batch, x_colloc, epoch, compute_pde=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss_total / num_batches
        
        if epoch % 500 == 0:
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
        
        # Save latest checkpoint every 500 epochs
        if epoch % 500 == 0 and epoch > 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss_history': loss_history
            }, checkpoint_path)
    
    total_time = time.time() - start_time
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/reaction_loss.json", 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # Save timing info
    timing_info = {
        'stage': 'reaction',
        'epochs_trained': epochs - start_epoch,
        'time_seconds': total_time,
        'time_minutes': total_time / 60,
        'time_hours': total_time / 3600
    }
    with open(f"{config.RESULTS_DIR}/reaction_timing.json", 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    # Save final model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/reaction_final.pt")
    
    print(f"\nReaction training complete!")
    print(f"Time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Loss history saved to {config.RESULTS_DIR}/reaction_loss.json")
    print(f"Timing saved to {config.RESULTS_DIR}/reaction_timing.json")
    
    return model