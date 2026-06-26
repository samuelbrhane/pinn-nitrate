import numpy as np
import pandas as pd
import os

def generate_benchmark(bench_num, n_spatial=100, n_temporal=100):
    """Generate synthetic benchmark data"""
    
    x = np.linspace(0, 100, n_spatial)
    t = np.linspace(0, 100, n_temporal)
    u = 0.5
    
    data = []
    
    for t_val in t:
        for x_val in x:
            # Adjust complexity by benchmark
            if bench_num == 1:
                # Linear denitrification
                NO3 = 50 * np.exp(-0.01 * (x_val/u + t_val))
                DOC = 200 * np.exp(-0.015 * (x_val/u + t_val))
                Fe2 = 1 + 5 * (1 - np.exp(-0.008 * (x_val/u + t_val)))
                
            elif bench_num == 2:
                # Dual competition
                NO3 = 50 * np.exp(-0.012 * (x_val/u + t_val))
                DOC = 200 * np.exp(-0.018 * (x_val/u + t_val))
                Fe2 = 1 + 8 * (1 - np.exp(-0.01 * (x_val/u + t_val)))
                
            elif bench_num == 3:
                # Monod kinetics
                NO3 = 50 * np.exp(-0.015 * (x_val/u + t_val))
                DOC = 200 * np.exp(-0.02 * (x_val/u + t_val))
                Fe2 = 1 + 10 * (1 - np.exp(-0.012 * (x_val/u + t_val)))
                
            else: 
                # Complex dual Monod
                NO3 = 50 * np.exp(-0.018 * (x_val/u + t_val))
                DOC = 200 * np.exp(-0.025 * (x_val/u + t_val))
                Fe2 = 1 + 12 * (1 - np.exp(-0.015 * (x_val/u + t_val)))
            
            N2 = 50 - NO3
            NO3 = max(0, NO3)
            
            data.append({
                'x': x_val, 't': t_val, 'u': u,
                'NO3': NO3, 'DOC': DOC, 'Fe2+': Fe2, 'N2': N2
            })
    
    df = pd.DataFrame(data)
    filename = f'data/synthetic/benchmark{bench_num}.csv'
    df.to_csv(filename, index=False)
    print(f"Benchmark {bench_num}: {len(df)} points → {filename}")
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    # Generate all 4 benchmarks
    for i in range(1, 5):
        generate_benchmark(i)
    
    print("\nAll benchmarks generated!")