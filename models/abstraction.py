from abc import ABC, abstractmethod
from typing import List, Dict
import torch
import torch.nn as nn

class IBaseNeuralNetworkModel(ABC):

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        layers = []
        current_input_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, size))
            layers.append(nn.ReLU())
            current_input_size = size
        layers.append(nn.Linear(current_input_size, output_size))
        self.model = nn.Sequential(*layers)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()



    @abstractmethod
    def train(self, x_train: torch.Tensor, y_train: torch.Tensor,
              x_val: torch.Tensor = None, y_val: torch.Tensor = None,
              max_nfe: int = 15000, patience: int = 10) -> list:
        # Returns loss history
        pass

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
        return outputs

    def save(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))