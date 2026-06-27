import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess

def run_command(cmd, description):
    """Run command and report status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR in {description}!")
        return False
    print(f"{description} complete!")
    return True

def run_benchmark(benchmark_num):
    """Run all stages for one benchmark, skip if already done"""
    print(f"\n\n{'='*60}")
    print(f"BENCHMARK {benchmark_num}")
    print(f"{'='*60}\n")
    
    # Update config.BENCHMARK
    with open('config.py', 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('BENCHMARK = '):
            lines[i] = f'BENCHMARK = {benchmark_num}\n'
            break
    
    with open('config.py', 'w') as f:
        f.writelines(lines)
    
    print(f"Updated config: BENCHMARK = {benchmark_num}\n")
    
    # Training stages (skip if final model exists in benchmark folder)
    benchmark_model_dir = f"models/benchmark{benchmark_num}"
    training_stages = [
        ('python scripts/run_transport.py', 'Transport (Stage 1)', f'{benchmark_model_dir}/transport_final.pt'),
        ('python scripts/run_reaction.py', 'Reaction (Stage 2)', f'{benchmark_model_dir}/reaction_final.pt'),
        ('python scripts/run_finetune.py', 'Fine-tune (Stage 3)', f'{benchmark_model_dir}/finetune_final.pt'),
        ('python scripts/run_baseline.py', 'Baseline Training', f'{benchmark_model_dir}/baseline_final.pt'),
    ]
    
    for cmd, desc, final_model in training_stages:
        if os.path.exists(final_model):
            print(f"{desc} already complete, skipping...")
            continue
        if not run_command(cmd, desc):
            print(f"\nStopped at {desc}. Re-run to continue from checkpoint.")
            return False
    
    # Always run (safe to rerun - just read and output results)
    run_command('python scripts/run_evaluate.py', 'Evaluate ST-PINN')
    run_command('python scripts/plot_loss_history.py', 'Plot Loss History')
    
    return True

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"MASTER TRAINING: All 4 Benchmarks")
    print(f"{'='*60}\n")
    
    success_count = 0
    
    for benchmark in range(1, 2):
        if not run_benchmark(benchmark):
            print(f"\nStopped at Benchmark {benchmark}. Re-run script to continue.\n")
            break
        success_count += 1
    
    if success_count == 4:
        print(f"\n{'='*60}")
        print(f"ALL 4 BENCHMARKS COMPLETE!")
        print(f"{'='*60}\n")
    else:
        print(f"\nCompleted: {success_count}/4 benchmarks\n")