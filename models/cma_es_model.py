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

    def train(self, x_train: torch.Tensor, y_train: torch.Tensor, iterations: int) -> list:
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

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

        for generation in range(iterations):
            population = self.cmaes.ask()

            fitnesses = []

            for i in range(self.population_size):
                torch.nn.utils.vector_to_parameters(population[i], self.model.parameters())

                outputs = self.model(x_train)
                loss = self.criterion(outputs, y_train)
                fitnesses.append(loss.item())

            self.cmaes.tell(fitnesses)

            best_loss = min(fitnesses)
            loss_history.append(best_loss)

            if (generation + 1) % 10 == 0 or generation == 0:
                print(f"Generation {generation+1:4d} | Best Loss: {best_loss:.6f} | Sigma: {self.cmaes.sigma:.6f}")

        torch.nn.utils.vector_to_parameters(self.cmaes.m, self.model.parameters())
        print("Training Complete. Model parameters updated to the final mean.")

        return loss_history

        