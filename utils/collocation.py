import torch
import numpy as np
import config

class CollocationPointGenerator:
    """Generate random collocation points for PDE enforcement"""
    
    def __init__(self, n_collocation=1200, domain_x=(0, 100), domain_t=(0, 100), velocity=0.5):
        self.n_collocation = n_collocation
        self.domain_x = domain_x
        self.domain_t = domain_t
        self.velocity = velocity
    
    def generate(self):
        """Generate random collocation points"""
        x = np.random.uniform(self.domain_x[0], self.domain_x[1], self.n_collocation)
        t = np.random.uniform(self.domain_t[0], self.domain_t[1], self.n_collocation)
        u = np.ones(self.n_collocation) * self.velocity
        
        colloc = np.column_stack([x, t, u])
        return torch.tensor(colloc, dtype=torch.float32, device=config.DEVICE)