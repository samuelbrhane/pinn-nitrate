import torch
import torch.nn as nn

class DynamicLossWeighting:
    def __init__(self):
        self.weights = {
            'data': 1.0,
            'pde': 1.0,
            'ic': 1.0,
            'bc': 1.0
        }
    
    def update_weights(self, epoch, loss_dict):
        """Dynamically adjust weights every 100 epochs"""
        if epoch % 100 == 0 and epoch > 0:
            # Normalize losses
            max_loss = max(loss_dict.values())
            if max_loss > 0:
                for key in self.weights:
                    if key in loss_dict:
                        # Inverse of loss: smaller loss gets higher weight
                        self.weights[key] = max_loss / (loss_dict[key] + 1e-6)
            
            # Normalize weights to sum to ~4
            total = sum(self.weights.values())
            for key in self.weights:
                self.weights[key] = (self.weights[key] / total) * 4.0

class PINNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dlw = DynamicLossWeighting()
        self.mse = nn.MSELoss()
    
    def data_loss(self, y_pred, y_true):
        """Loss on PHREEQC data points"""
        return self.mse(y_pred, y_true)
    
    def pde_residual(self, y_pred):
        """PDE residual (simplified)"""
        # Placeholder: actual PDE would go here
        # For now, regularization term
        return torch.mean(y_pred ** 2) * 0.01
    
    def ic_loss(self, y_pred_ic, y_true_ic):
        """Initial condition loss (at t=0)"""
        return self.mse(y_pred_ic, y_true_ic)
    
    def bc_loss(self, y_pred_bc, y_true_bc):
        """Boundary condition loss (at x=0, x=100)"""
        return self.mse(y_pred_bc, y_true_bc)
    
    def forward(self, y_pred, y_true, epoch, is_stage1=False):
        """Compute total loss with DLW"""
        
        loss_data = self.data_loss(y_pred, y_true)
        loss_pde = self.pde_residual(y_pred)
        loss_ic = self.ic_loss(y_pred[:100], y_true[:100])  # First 100 points (t=0)
        loss_bc = self.bc_loss(y_pred[::100], y_true[::100])  # Every 100th point (boundaries)
        
        # Collect losses
        loss_dict = {
            'data': loss_data.item(),
            'pde': loss_pde.item(),
            'ic': loss_ic.item(),
            'bc': loss_bc.item()
        }
        
        # Update weights every 100 epochs
        self.dlw.update_weights(epoch, loss_dict)
        
        # Weighted sum
        total_loss = (
            self.dlw.weights['data'] * loss_data +
            self.dlw.weights['pde'] * loss_pde +
            self.dlw.weights['ic'] * loss_ic +
            self.dlw.weights['bc'] * loss_bc
        )
        
        return total_loss, loss_dict