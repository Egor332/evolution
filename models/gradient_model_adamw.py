import torch
import torch.nn as nn
from typing import Dict, List, Any 
from models.abstraction import IBaseNeuralNetworkModel

class GradientModelAdamW(IBaseNeuralNetworkModel):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, lr: float = 0.001):
        super().__init__(input_size, hidden_sizes, output_size)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def train(self, x_train: torch.Tensor, y_train: torch.Tensor, iterations: int = 100) -> list: 
       loss_history = []
       x_train = x_train.to(self.device)
       y_train = y_train.to(self.device)
       self.model.to(self.device)

       for iteration in range(iterations):
           self.optimizer.zero_grad()
           outputs = self.model(x_train)
           loss = self.criterion(outputs, y_train)
           loss.backward()
           self.optimizer.step()
           loss_history.append(loss.item())
       
       return loss_history