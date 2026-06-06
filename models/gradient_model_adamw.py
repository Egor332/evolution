import torch
import torch.nn as nn
from typing import Dict, List, Any 
from models.abstraction import IBaseNeuralNetworkModel

class GradientModelAdamW(IBaseNeuralNetworkModel):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, lr: float = 0.001):
        super().__init__(input_size, hidden_sizes, output_size)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr)


    def train(self, x_train: torch.Tensor, y_train: torch.Tensor, 
              x_val: torch.Tensor = None, y_val: torch.Tensor = None, 
              max_nfe: int = 15000, patience: int = 10) -> list: 
       loss_history = []
       x_train = x_train.to(self.device)
       y_train = y_train.to(self.device)
       if x_val is not None and y_val is not None:
           x_val = x_val.to(self.device)
           y_val = y_val.to(self.device)
       self.model.to(self.device)

       best_val_loss = float('inf')
       best_weights = None
       nfe_no_improve = 0

       for iteration in range(max_nfe):
           self.model.train()
           self.optimizer.zero_grad()
           outputs = self.model(x_train)
           loss = self.criterion(outputs, y_train)
           loss.backward()
           self.optimizer.step()
           loss_history.append(loss.item())
           
           if x_val is not None and y_val is not None:
               self.model.eval()
               with torch.no_grad():
                   val_outputs = self.model(x_val)
                   val_loss = self.criterion(val_outputs, y_val).item()
                   
               if val_loss < best_val_loss:
                   best_val_loss = val_loss
                   best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                   nfe_no_improve = 0
               else:
                   nfe_no_improve += 1
                   
               if nfe_no_improve >= patience:
                   print(f"Early stopping at iteration {iteration} (Best Val Loss: {best_val_loss:.4f})")
                   self.model.load_state_dict(best_weights)
                   self.model.to(self.device)
                   break
       
       return loss_history