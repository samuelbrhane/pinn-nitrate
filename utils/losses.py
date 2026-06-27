import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import config


class DynamicLossWeighting:
    def __init__(self, stage='transport'):
        if stage == 'transport':
            self.weights = {'data': 10.0, 'pde': 1.0, 'ic': 1.0, 'bc': 1.0}
        elif stage == 'reaction':
            self.weights = {'data': 10.0, 'pde': 10.0, 'ic': 1.0, 'bc': 1.0}
        elif stage == 'finetune':
            self.weights = {'data': 10.0, 'pde': 1.0, 'ic': 1.0, 'bc': 1.0}
        else:
            self.weights = {'data': 5.0, 'pde': 1.0, 'ic': 1.0, 'bc': 1.0}
        
        self.initial_weights = self.weights.copy()
    
    def update_weights(self, epoch, loss_dict):
        if epoch % config.LOSS_WEIGHT_UPDATE_FREQ == 0 and epoch > 0:
            losses = [max(loss_dict.get(key, 0.0), 1e-8) for key in self.weights]
            max_loss = max(losses)
            
            new_weights = {key: max_loss / loss for key, loss in zip(self.weights.keys(), losses)}
            
            total_new = sum(new_weights.values())
            total_old = sum(self.initial_weights.values())
            scale = total_old / (total_new + 1e-8)
            
            for key in new_weights:
                new_weights[key] = np.clip(new_weights[key] * scale, 0.1, 100.0)
                
                if key == 'data':
                    new_weights[key] = max(new_weights[key], 5.0)
                elif key in ['ic', 'bc']:
                    new_weights[key] = max(new_weights[key], 1.0)
                elif key == 'pde':
                    new_weights[key] = max(new_weights[key], 0.5)
            
            self.weights = new_weights


class MultiSpeciesPINNLoss(nn.Module):
    def __init__(self, model, device, D=1.0, k=0.01, u=0.5, stage='transport'):
        super().__init__()
        self.dlw = DynamicLossWeighting(stage)
        self.mse = nn.MSELoss()
        self.model = model
        self.device = device
        self.D = D
        self.k = k
        self.u = u
        
        self.stoich = {
            'NO3': config.STOICH_NO3,
            'DOC': config.STOICH_DOC,
            'Fe2': config.STOICH_FE2,
            'N2': config.STOICH_N2
        }
        
        self.species_indices = {
            'NO3': 0,
            'DOC': 1,
            'Fe2': 2,
            'N2': 3
        }
    
    def data_loss(self, y_pred, y_true):
        return self.mse(y_pred, y_true)
    
    def pde_residual(self, x_colloc):
        if len(x_colloc) == 0:
            return torch.tensor(0.0, device=self.device)
        
        x = x_colloc.clone().detach().requires_grad_(True)
        c_all = self.model(x)
        
        c_NO3 = c_all[:, 0:1]
        c_DOC = c_all[:, 1:2]
        c_Fe2 = c_all[:, 2:3]
        c_N2 = c_all[:, 3:4]
        species = [c_NO3, c_DOC, c_Fe2, c_N2]
        species_names = ['NO3', 'DOC', 'Fe2', 'N2']
        
        pde_residuals = []
        
        for c, name in zip(species, species_names):
            grad_c = torch.autograd.grad(
                c, x,
                grad_outputs=torch.ones_like(c),
                create_graph=True,
                retain_graph=True
            )[0]
            
            dc_dx = grad_c[:, 0:1]
            dc_dt = grad_c[:, 1:2]
            
            d2c_dx2 = torch.autograd.grad(
                dc_dx, x,
                grad_outputs=torch.ones_like(dc_dx),
                create_graph=True,
                retain_graph=True
            )[0][:, 0:1]
            
            if name == 'NO3':
                R = -self.k * c_NO3
            elif name == 'DOC':
                R = self.stoich['DOC'] * self.k * c_NO3
            elif name == 'Fe2':
                R = self.stoich['Fe2'] * self.k * c_NO3
            elif name == 'N2':
                R = self.stoich['N2'] * self.k * c_NO3
            else:
                R = torch.zeros_like(c)
            
            u = x[:, 2:3]
            pde = dc_dt + u * dc_dx - self.D * d2c_dx2 - R
            pde_residuals.append(torch.mean(pde ** 2))
        
        return torch.mean(torch.stack(pde_residuals))
    
    def ic_loss(self, x_ic, c_ic_true, u=0.5):
        t_ic = torch.zeros_like(x_ic)
        u_ic = u * torch.ones_like(x_ic)
        x_input = torch.cat([x_ic, t_ic, u_ic], dim=1)
        c_ic_pred = self.model(x_input)
        
        return self.mse(c_ic_pred, c_ic_true)
    
    def bc_loss(self, t_bc, c_bc_true, x_boundary=0.0, u=0.5):
        x_bc = x_boundary * torch.ones_like(t_bc)
        u_bc = u * torch.ones_like(t_bc)
        x_input = torch.cat([x_bc, t_bc, u_bc], dim=1)
        c_bc_pred = self.model(x_input)
        
        return self.mse(c_bc_pred, c_bc_true)
    
    def forward(self, x_data, y_pred_data, y_true_data, x_colloc,
                x_ic, c_ic_true, t_bc, c_bc_true, epoch,
                compute_pde=True):
        loss_data = self.data_loss(y_pred_data, y_true_data)
        
        if compute_pde:
            loss_pde = self.pde_residual(x_colloc)
        else:
            loss_pde = torch.tensor(0.0, device=self.device)
        
        loss_ic = self.ic_loss(x_ic, c_ic_true, u=self.u)
        loss_bc = self.bc_loss(t_bc, c_bc_true, x_boundary=0.0, u=self.u)
        
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