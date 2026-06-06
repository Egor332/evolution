from models.abstraction import IBaseNeuralNetworkModel
from typing import List
import torch
import math
from models.scratch_cma_es import ScratchCMAES

class CMAESModel(IBaseNeuralNetworkModel):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, sigma0: float, population_size: int = 100):
        super().__init__(input_size, hidden_sizes, output_size)

        self.sigma0 = sigma0
        self.population_size = population_size
        
        dummy_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = dummy_params.numel()

        # STANDARD CMA-ES HEURISTICS 
        self.parents_size = self.population_size // 2
        
        # Calculate mu_eff to derive the learning rates
        weights = math.log(self.parents_size + 0.5) - torch.log(torch.arange(1, self.parents_size + 1))
        weights = weights / weights.sum()
        self.mu_eff = (1.0 / (weights ** 2).sum()).item()
        
        # Standard learning rates for CMA-ES
        self.c_c = (4 + self.mu_eff / self.N) / (self.N + 4 + 2 * self.mu_eff / self.N)
        self.c_sigma = (self.mu_eff + 2) / (self.N + self.mu_eff + 5)
        self.c_1 = 2 / ((self.N + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.N + 2)**2 + self.mu_eff))
        
        # Damping factor
        self.d_sigma = 1.0 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (self.N + 1)) - 1) + self.c_sigma

    def train(self, x_train: torch.Tensor, y_train: torch.Tensor, 
              x_val: torch.Tensor = None, y_val: torch.Tensor = None, 
              max_nfe: int = 15000, patience: int = 10) -> list:
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        if x_val is not None and y_val is not None:
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

        initial_params = torch.nn.utils.parameters_to_vector(self.model.parameters())

        self.cmaes = ScratchCMAES(
            num_params=self.N,
            population_size=self.population_size,
            parents_size=self.parents_size,
            sigma=self.sigma0,
            c_c=self.c_c,
            c_sigma=self.c_sigma,
            d_sigma=self.d_sigma,
            c_1=self.c_1,
            c_mu=self.c_mu,
            mu_eff=self.mu_eff,
            device=self.device
        )

        self.cmaes.m = initial_params.clone()

        print(f"Starting CMA-ES Training. Parameters (N): {self.N} | Population: {self.population_size}")

        loss_history = []
        nfe = 0
        best_val_loss = float('inf')
        best_weights_m = None
        nfe_no_improve = 0

        generation = 0
        while nfe < max_nfe:
            population = self.cmaes.ask()

            fitnesses = []

            for i in range(self.population_size):
                if nfe >= max_nfe:
                    break
                torch.nn.utils.vector_to_parameters(population[i], self.model.parameters())

                outputs = self.model(x_train)
                loss = self.criterion(outputs, y_train)
                fitnesses.append(loss.item())
                nfe += 1

            self.cmaes.tell(fitnesses)

            best_loss = min(fitnesses)
            loss_history.append(best_loss)

            if (generation + 1) % 10 == 0 or generation == 0:
                print(f"Generation {generation+1:4d} | Best Loss: {best_loss:.6f} | Sigma: {self.cmaes.sigma:.6f}")

            if x_val is not None and y_val is not None:
                # evaluate validation on current mean
                torch.nn.utils.vector_to_parameters(self.cmaes.m, self.model.parameters())
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(x_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights_m = self.cmaes.m.clone()
                    nfe_no_improve = 0
                else:
                    nfe_no_improve += self.population_size
                
                if nfe_no_improve >= patience:
                    print(f"Early stopping at generation {generation} (NFE: {nfe}, Best Val Loss: {best_val_loss:.4f})")
                    if best_weights_m is not None:
                        self.cmaes.m = best_weights_m
                    break
                    
            generation += 1

        torch.nn.utils.vector_to_parameters(self.cmaes.m, self.model.parameters())
        print(f"Training Complete. NFE: {nfe}. Model parameters updated to the final mean.")

        return loss_history

        