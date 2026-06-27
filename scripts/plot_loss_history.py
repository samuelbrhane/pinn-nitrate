import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import config

sns.set_style('whitegrid')

def plot_transport_loss():
    """Plot transport (Stage 1) loss convergence"""
    loss_file = f"{config.RESULTS_DIR}/transport_loss.json"
    if not os.path.exists(loss_file):
        print(f"transport_loss.json not found, skipping")
        return
    
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.semilogy(epochs, val_loss, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Stage 1: Transport Training Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{config.PLOTS_DIR}/transport_convergence.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/transport_convergence.png")
    plt.close()

def plot_reaction_loss():
    """Plot reaction (Stage 2) loss convergence"""
    loss_file = f"{config.RESULTS_DIR}/reaction_loss.json"
    if not os.path.exists(loss_file):
        print(f"reaction_loss.json not found, skipping")
        return
    
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.semilogy(epochs, val_loss, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Stage 2: Reaction Training Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{config.PLOTS_DIR}/reaction_convergence.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/reaction_convergence.png")
    plt.close()

def plot_finetune_loss():
    """Plot fine-tune (Stage 3) loss convergence"""
    loss_file = f"{config.RESULTS_DIR}/finetune_loss.json"
    if not os.path.exists(loss_file):
        print(f"finetune_loss.json not found, skipping")
        return
    
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.semilogy(epochs, val_loss, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Stage 3: Fine-tuning Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{config.PLOTS_DIR}/finetune_convergence.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/finetune_convergence.png")
    plt.close()

def plot_baseline_loss():
    """Plot baseline (standard PINN) loss convergence"""
    loss_file = f"{config.RESULTS_DIR}/baseline_loss.json"
    if not os.path.exists(loss_file):
        print(f"baseline_loss.json not found, skipping")
        return
    
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.semilogy(epochs, val_loss, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Baseline: Standard PINN Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{config.PLOTS_DIR}/baseline_convergence.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/baseline_convergence.png")
    plt.close()

def plot_all_stages():
    """Plot all 3 ST-PINN stages in one figure"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    stages = [
        (f"{config.RESULTS_DIR}/transport_loss.json", 'Transport', axes[0]),
        (f"{config.RESULTS_DIR}/reaction_loss.json", 'Reaction', axes[1]),
        (f"{config.RESULTS_DIR}/finetune_loss.json", 'Fine-tune', axes[2])
    ]
    
    for loss_file, stage_name, ax in stages:
        if not os.path.exists(loss_file):
            print(f"{loss_file} not found, skipping")
            continue
        
        with open(loss_file, 'r') as f:
            data = json.load(f)
        
        epochs = data['epochs']
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        
        ax.semilogy(epochs, train_loss, label='Train', linewidth=2)
        ax.semilogy(epochs, val_loss, label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'Stage: {stage_name}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{config.PLOTS_DIR}/all_stages_convergence.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/all_stages_convergence.png")
    plt.close()

def plot_stpinn_vs_baseline():
    """Plot ST-PINN finetune vs Baseline comparison"""
    stpinn_file = f"{config.RESULTS_DIR}/finetune_loss.json"
    baseline_file = f"{config.RESULTS_DIR}/baseline_loss.json"
    
    if not os.path.exists(stpinn_file) or not os.path.exists(baseline_file):
        print(f"ST-PINN or Baseline loss files not found, skipping comparison")
        return
    
    with open(stpinn_file, 'r') as f:
        stpinn_data = json.load(f)
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(stpinn_data['epochs'], stpinn_data['val_loss'], label='ST-PINN', linewidth=2.5, marker='o', markersize=4)
    plt.semilogy(baseline_data['epochs'], baseline_data['val_loss'], label='Baseline', linewidth=2.5, marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (log scale)')
    plt.title('ST-PINN vs Baseline: Convergence Comparison')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{config.PLOTS_DIR}/stpinn_vs_baseline.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.PLOTS_DIR}/stpinn_vs_baseline.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    print(f"\nPlotting loss history for benchmark {config.BENCHMARK}...\n")
    
    plot_transport_loss()
    plot_reaction_loss()
    plot_finetune_loss()
    plot_baseline_loss()
    plot_all_stages()
    plot_stpinn_vs_baseline()
    
    print(f"\nAll loss plots saved to {config.PLOTS_DIR}/")