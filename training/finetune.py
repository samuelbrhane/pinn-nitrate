import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import config
import json

def train_finetune(model, train_loader, val_loader, loss_fn, epochs=10000):
    """Fine-tune Training: Unfreeze all weights with checkpoint resume"""
    print(f"\n{'='*60}")
    print(f"Fine-tune Training (10k epochs)")
    print(f"{'='*60}\n")
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
    loss_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # Check for checkpoint
    checkpoint_path = f"{config.MODEL_DIR}/finetune_latest.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss_history = checkpoint['loss_history']
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
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
        
        # Save latest checkpoint every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss_history': loss_history
            }, checkpoint_path)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/finetune_loss.json", 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # Save final model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/finetune_final.pt")
    
    print(f"\nFine-tune training complete!")
    print(f"Loss history saved to {config.RESULTS_DIR}/finetune_loss.json")
    
    return model