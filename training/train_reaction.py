import torch
import torch.optim as optim
import config
import os

def train_reaction(model, train_loader, val_loader, loss_fn, epochs=20000):
    """Stage 2: Train reaction component (freeze transport)"""
    print(f"\n{'='*60}")
    print(f"Reaction Training (20k epochs)")
    print(f"{'='*60}\n")
    
    # Freeze transport weights
    for name, param in model.named_parameters():
        if 'network.0' in name or 'network.1' in name:
            param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(epochs):
        train_loss_total = 0
        for X_batch, Y_batch in train_loader:
            y_pred = model(X_batch)
            loss, loss_dict = loss_fn(y_pred, Y_batch, epoch, is_stage1=False)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
        
        if epoch % 100 == 0:
            val_loss_total = 0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    y_pred_val = model(X_val)
                    loss_val, _ = loss_fn(y_pred_val, Y_val, epoch, is_stage1=False)
                    val_loss_total += loss_val.item()
            
            avg_train_loss = train_loss_total / len(train_loader)
            avg_val_loss = val_loss_total / len(val_loader)
            
            print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if epoch % config.CHECKPOINT_FREQ == 0 and epoch > 0:
            torch.save(model.state_dict(), f"{config.MODEL_DIR}/reaction_epoch{epoch}.pt")
    
    print(f"\nReaction complete!")
    return model