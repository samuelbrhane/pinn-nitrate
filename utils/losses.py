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
        """Dynamically adjust weights every 100 epochs to balance loss components"""
        if epoch % 100 == 0 and epoch > 0:
            max_loss = max(loss_dict.values())
            if max_loss > 0:
                for key in self.weights:
                    if key in loss_dict:
                        self.weights[key] = max_loss / (loss_dict[key] + 1e-6)
            
            total = sum(self.weights.values())
            for key in self.weights:
                self.weights[key] = (self.weights[key] / total) * 4.0

class PINNLoss(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.dlw = DynamicLossWeighting()
        self.mse = nn.MSELoss()
        self.model = model
        self.device = device
        self.D = 0.1
    
    def data_loss(self, y_pred, y_true):
        """Measure prediction error on training data points"""
        return self.mse(y_pred, y_true)
    
    def pde_residual(self, x):
        """Enforce reactive transport PDE: ∂c/∂t + u·∂c/∂x - D·∂²c/∂x² = 0"""
        x_req = x.clone().detach().requires_grad_(True)
        y = self.model(x_req)
        
        dy_dx = torch.autograd.grad(y.sum(), x_req, create_graph=True)[0]
        dc_dt = dy_dx[:, 1]
        dc_dx = dy_dx[:, 0]
        
        d2c_dx2 = torch.autograd.grad(dc_dx.sum(), x_req, create_graph=True)[0][:, 0]
        
        u = x[:, 2]
        pde = dc_dt + u * dc_dx - self.D * d2c_dx2
        
        return torch.mean(pde ** 2)
    
    def ic_loss(self, y_pred_ic, y_true_ic):
        """Enforce initial conditions at t=0"""
        return self.mse(y_pred_ic, y_true_ic)
    
    def bc_loss(self, y_pred_bc, y_true_bc):
        """Enforce boundary conditions at x=0 and x=100"""
        return self.mse(y_pred_bc, y_true_bc)
    
    def forward(self, x_data, y_pred_data, y_true_data, x_colloc, epoch, compute_pde=True):
        """Compute loss on data + collocation points"""
        loss_data = self.data_loss(y_pred_data, y_true_data)
        
        if compute_pde:
            loss_pde = self.pde_residual(x_colloc)
        else:
            loss_pde = torch.tensor(0.0, device=self.device)
        
        loss_ic = self.ic_loss(y_pred_data[:100], y_true_data[:100])
        loss_bc = self.ic_loss(y_pred_data[::100], y_true_data[::100])
        
        loss_dict = {
            'data': loss_data.item(),
            'pde': loss_pde.item() if compute_pde else 0.0,
            'ic': loss_ic.item(),
            'bc': loss_bc.item()
        }
        
        self.dlw.update_weights(epoch, loss_dict)
        
        total_loss = (
            self.dlw.weights['data'] * loss_data +
            self.dlw.weights['pde'] * loss_pde +
            self.dlw.weights['ic'] * loss_ic +
            self.dlw.weights['bc'] * loss_bc
        )
        
        return total_loss, loss_dict