import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import config

sns.set_style('whitegrid')

def plot_stage_loss(stage_name):
    """Plot loss convergence for a specific stage"""
    loss_file = f"{config.RESULTS_DIR}/{stage_name}_loss.json"
    
    if not os.path.exists(loss_file):
        print(f"{stage_name}_loss.json not found")
        return False
    
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    
    titles = {
        'transport': 'Stage 1: Transport Training Convergence',
        'reaction': 'Stage 2: Reaction Training Convergence',
        'finetune': 'Stage 3: Fine-tuning Convergence',
        'baseline': 'Baseline: Standard PINN Training (Comparison)'
    }
    
    # Different color for baseline
    if stage_name == 'baseline':
        train_color = '#d62728'  # Red
        val_color = '#ff7f0e'    # Orange
    else:
        train_color = '#1f77b4'  # Blue
        val_color = '#ff7f0e'    # Orange
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_loss, label='Train Loss', linewidth=2.5, color=train_color)
    plt.semilogy(epochs, val_loss, label='Val Loss', linewidth=2.5, color=val_color)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (log scale)', fontsize=11)
    plt.title(titles.get(stage_name, f'{stage_name.capitalize()} Training'), fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_file = f"{config.PLOTS_DIR}/{stage_name}_convergence.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_single_stage.py [stage]")
        print("Stages: transport, reaction, finetune, baseline")
        print("Example: python plot_single_stage.py transport")
        sys.exit(1)
    
    stage = sys.argv[1].lower()
    valid_stages = ['transport', 'reaction', 'finetune', 'baseline']
    
    if stage not in valid_stages:
        print(f"Invalid stage: {stage}")
        print(f"Valid stages: {', '.join(valid_stages)}")
        sys.exit(1)
    
    print(f"Plotting {stage} loss for benchmark {config.BENCHMARK}...\n")
    
    if plot_stage_loss(stage):
        print(f"\n{stage.capitalize()} loss plot created!")
    else:
        print(f"\nFailed to plot {stage} loss")