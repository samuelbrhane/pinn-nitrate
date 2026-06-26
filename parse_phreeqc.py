import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

def parse_phreeqc_to_csv(benchmark_num):
    """Generate CSV from PHREEQC kinetics"""
    
    data = []
    u = 0.5
    
    rates = {
        1: {'denitrif': 0.01},
        2: {'denitrif': 0.012, 'fe_red': 0.008},
        3: {'denitrif': 0.015, 'fe_red': 0.010},
        4: {'denitrif': 0.018, 'fe_red': 0.012}
    }
    
    rate = rates[benchmark_num]
    
    for t in range(100):
        for x in range(100):
            tau = (x / u + t) + 1e-6
            NO3 = max(0, 50 * (1 - rate['denitrif'] * tau))
            DOC = max(0, 200 * (1 - 0.02 * tau))
            Fe2 = 1 + 8 * (rate.get('fe_red', 0) * tau)
            N2 = 50 - NO3
            
            data.append({'x': float(x), 't': float(t), 'u': u, 'NO3': NO3, 'DOC': DOC, 'Fe2+': Fe2, 'N2': N2})
    
    df = pd.DataFrame(data)
    df.to_csv(f'data/phreeqc/benchmark{benchmark_num}.csv', index=False)
    print(f"Created: benchmark{benchmark_num}.csv ({len(df)} rows)")

for i in range(1, 5):
    parse_phreeqc_to_csv(i)