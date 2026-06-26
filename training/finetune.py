import torch
import torch.optim as optim
import config
import os
import json

def train_finetune(model, train_loader, val_loader, loss_fn, epochs=10000):
    """Fine-tune Training: Unfreeze all weights and optimize together"""
    print(f"\n{'='*60}")
    print(f"Fine-tune Training (10k epochs)")
    print(f"{'='*60}\n")
    
    # Unfreeze all weights
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Track losses for plotting
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
            loss, loss_dict = loss_fn(y_pred, Y_batch, epoch, is_stage1=False)
            
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
                    loss_val, _ = loss_fn(y_pred_val, Y_val, epoch, is_stage1=False)
                    val_loss_total += loss_val.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss_total / num_val_batches
            
            # Record loss
            loss_history['epochs'].append(epoch)
            loss_history['train_loss'].append(avg_train_loss)
            loss_history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if epoch % config.CHECKPOINT_FREQ == 0 and epoch > 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.MODEL_DIR}/finetune_epoch{epoch}.pt")
    
    # Save loss history
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/finetune_loss.json", 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"\nFine-tune training complete!")
    print(f"Loss history saved to {config.RESULTS_DIR}/finetune_loss.json")
    
    return model