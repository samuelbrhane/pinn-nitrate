import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=4, hidden_dim=128, num_layers=6):
        super().__init__()
        
        # Build network: 3 → 128 → 128 → 128 → 128 → 128 → 128 → 4
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)