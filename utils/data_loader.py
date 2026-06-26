import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import config
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

class DataHandler:
    def __init__(self, benchmark_num=1):
        """Load PHREEQC data"""
        self.benchmark_num = benchmark_num
        self.filepath = f"data/phreeqc/benchmark{benchmark_num}.csv"
        self.scaler_X = MinMaxScaler()
        self.scaler_Y = MinMaxScaler()
    
    def load_and_normalize(self):
        """Load data and normalize"""
        df = pd.read_csv(self.filepath)
        print(f"Loaded {len(df)} points from benchmark {self.benchmark_num}")
        
        # Input: x, t, u
        X = df[['x', 't', 'u']].values
        
        # Output: NO3, DOC, Fe2+, N2
        Y = df[['NO3', 'DOC', 'Fe2+', 'N2']].values
        
        # Normalize
        X_norm = self.scaler_X.fit_transform(X)
        Y_norm = self.scaler_Y.fit_transform(Y)
        
        return X_norm, Y_norm
    
    def create_dataloaders(self):
        """Create train/val/test dataloaders"""
        X_norm, Y_norm = self.load_and_normalize()
        
        n = len(X_norm)
        n_train = int(config.TRAIN_SPLIT * n)
        n_val = int(config.VAL_SPLIT * n)
        
        # Split
        X_train = X_norm[:n_train]
        Y_train = Y_norm[:n_train]
        
        X_val = X_norm[n_train:n_train+n_val]
        Y_val = Y_norm[n_train:n_train+n_val]
        
        X_test = X_norm[n_train+n_val:]
        Y_test = Y_norm[n_train+n_val:]
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=config.DEVICE)
        Y_train = torch.tensor(Y_train, dtype=torch.float32, device=config.DEVICE)
        
        X_val = torch.tensor(X_val, dtype=torch.float32, device=config.DEVICE)
        Y_val = torch.tensor(Y_val, dtype=torch.float32, device=config.DEVICE)
        
        X_test = torch.tensor(X_test, dtype=torch.float32, device=config.DEVICE)
        Y_test = torch.tensor(Y_test, dtype=torch.float32, device=config.DEVICE)
        
        # Dataloaders
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return train_loader, val_loader, test_loader, self.scaler_X, self.scaler_Y
    
    def get_collocation_points(self, n_collocation=1200):
        """Get random collocation points"""
        from utils.collocation import CollocationPointGenerator
        gen = CollocationPointGenerator(n_collocation=n_collocation)
        return gen.generate()