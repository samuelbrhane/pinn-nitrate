# training/transport.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import config
from utils.losses import MultiSpeciesPINNLoss
import json
import time
from tqdm import tqdm

def train_transport(model, train_loader, val_loader, epochs=None):
    """
    Transport Training for multi-species reactive transport
    Input: [x, t, u]
    Output: [NO3, DOC, Fe2+, N2]
    
    🔥 Key improvements:
    - Fixed collocation points (same every epoch)
    - Weight minimums to prevent data/IC/BC from being ignored
    - Learning rate scheduler to prevent divergence
    - Early stopping with best model tracking
    """
    if epochs is None:
        epochs = config.EPOCHS_STAGE1
    
    print(f"\n{'='*70}")
    print(f"Transport Training ({epochs} epochs)")
    print(f"Input: [x, t, u], Output: [NO3, DOC, Fe2+, N2]")
    print(f"Physics: D={config.D}, k={config.K}, u={config.U}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Create loss function with correct parameters
    loss_fn = MultiSpeciesPINNLoss(
        model=model,
        device=config.DEVICE,
        D=config.D,
        k=config.K,
        u=config.U,
        stage='transport'
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 🔥 Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=500,
        min_lr=1e-7,
        verbose=True  # Set to True so you can see LR updates
    )
    
    start_epoch = 0
    loss_history = {'epochs': [], 'train_loss': [], 'val_loss': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    
    # Check for checkpoint
    checkpoint_path = os.path.join(config.MODEL_DIR, "transport_latest.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss_history = checkpoint['loss_history']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed from epoch {start_epoch}\n")
    
    # Prepare Initial Condition data (t=0, all x)
    x_ic = torch.linspace(
        config.X_MIN, config.X_MAX, 100, device=config.DEVICE
    ).reshape(-1, 1)
    c_ic_true = torch.cat([
        config.NO3_INLET * torch.ones_like(x_ic),      # NO3
        config.DOC_INLET * torch.ones_like(x_ic),      # DOC
        config.FE2_INLET * torch.ones_like(x_ic),      # Fe2+
        config.N2_INLET * torch.ones_like(x_ic)        # N2
    ], dim=1)
    
    # Prepare Boundary Condition data (x=0, all t)
    t_bc = torch.linspace(
        config.T_MIN, config.T_MAX, 100, device=config.DEVICE
    ).reshape(-1, 1)
    c_bc_true = torch.cat([
        config.NO3_INLET * torch.ones_like(t_bc),      # NO3
        config.DOC_INLET * torch.ones_like(t_bc),      # DOC
        config.FE2_INLET * torch.ones_like(t_bc),      # Fe2+
        config.N2_INLET * torch.ones_like(t_bc)        # N2
    ], dim=1)
    
    print(f"IC shape: {c_ic_true.shape}, BC shape: {c_bc_true.shape}")
    
    # Generate FIXED collocation points ONCE (BEFORE TRAINING LOOP)
    print("Generating fixed collocation points...")
    x_colloc = torch.rand(config.N_COLLOCATION, 1, device=config.DEVICE) * config.X_MAX
    t_colloc = torch.rand(config.N_COLLOCATION, 1, device=config.DEVICE) * config.T_MAX
    u_colloc = config.U * torch.ones(config.N_COLLOCATION, 1, device=config.DEVICE)
    x_colloc_fixed = torch.cat([x_colloc, t_colloc, u_colloc], dim=1)  # [N, 3]
    print(f"Fixed collocation points shape: {x_colloc_fixed.shape}\n")
    
    pbar = tqdm(range(start_epoch, epochs), desc="Training", initial=start_epoch, total=epochs)
    
    for epoch in pbar:
        # Use FIXED collocation points EVERY epoch
        x_colloc_input = x_colloc_fixed
        
        train_loss_total = 0.0
        num_batches = 0
        
        for X_batch, Y_batch in train_loader:
            y_pred = model(X_batch)
            
            loss, loss_dict = loss_fn(
                x_data=X_batch,
                y_pred_data=y_pred,
                y_true_data=Y_batch,
                x_colloc=x_colloc_input,
                x_ic=x_ic,
                c_ic_true=c_ic_true,
                t_bc=t_bc,
                c_bc_true=c_bc_true,
                epoch=epoch,
                compute_pde=True
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss_total / num_batches
        
        # Update progress bar
        pbar.set_postfix({'train_loss': f'{avg_train_loss:.6f}'})
        
        # Validation every 500 epochs
        if epoch % 500 == 0:
            val_loss_total = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    y_pred_val = model(X_val)
                    
                    dummy_colloc = torch.zeros(1, 3, device=config.DEVICE)
                    
                    loss_val, _ = loss_fn(
                        x_data=X_val,
                        y_pred_data=y_pred_val,
                        y_true_data=Y_val,
                        x_colloc=dummy_colloc,
                        x_ic=x_ic,
                        c_ic_true=c_ic_true,
                        t_bc=t_bc,
                        c_bc_true=c_bc_true,
                        epoch=epoch,
                        compute_pde=False
                    )
                    
                    val_loss_total += loss_val.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss_total / num_val_batches
            
            # 🔥 STEP THE SCHEDULER BASED ON VALIDATION LOSS
            scheduler.step(avg_val_loss)
            
            loss_history['epochs'].append(epoch)
            loss_history['train_loss'].append(avg_train_loss)
            loss_history['val_loss'].append(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                os.makedirs(config.MODEL_DIR, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "transport_best.pt"))
            else:
                patience_counter += 1
            
            print(f"\nEpoch {epoch}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Best: {best_val_loss:.6f}")
            print(f"  Loss components - Data: {loss_dict['data']:.4e}, PDE: {loss_dict['pde']:.4e}, "
                  f"IC: {loss_dict['ic']:.4e}, BC: {loss_dict['bc']:.4e}")
            print(f"  DLW Weights: {loss_fn.dlw.weights}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Early stopping patience: {patience_counter}/{patience_limit}\n")
            
            # Early stopping
            if patience_counter >= patience_limit:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        
        # Save checkpoint every 500 epochs
        if (epoch + 1) % 500 == 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            checkpoint_path = os.path.join(config.MODEL_DIR, "transport_latest.pt")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss_history': loss_history,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, checkpoint_path)
    
    # Load best model if early stopping was triggered
    best_model_path = os.path.join(config.MODEL_DIR, "transport_best.pt")
    if os.path.exists(best_model_path):
        print(f"\n📥 Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    
    total_time = time.time() - start_time
    
    # Save loss history
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "transport_loss.json"), 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # Save timing info
    timing_info = {
        'stage': 'transport',
        'epochs_trained': epochs - start_epoch,
        'best_val_loss': float(best_val_loss),
        'time_seconds': total_time,
        'time_minutes': total_time / 60,
        'time_hours': total_time / 3600
    }
    with open(os.path.join(config.RESULTS_DIR, "transport_timing.json"), 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    # Save final model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "transport_final.pt"))
    
    print(f"\n{'='*70}")
    print(f"✅ Transport training complete!")
    print(f"Time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Loss history saved to {config.RESULTS_DIR}/transport_loss.json")
    print(f"Timing saved to {config.RESULTS_DIR}/transport_timing.json")
    print(f"Final model saved to {config.MODEL_DIR}/transport_final.pt")
    print(f"Best model saved to {config.MODEL_DIR}/transport_best.pt")
    print(f"{'='*70}\n")
    
    return model