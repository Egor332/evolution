import torch
import math
from typing import List

class ScratchCMAES:
    def __init__(self, num_params: int, population_size: int,
        parents_size: int, sigma: float, 
        c_c: float, c_sigma: float, d_sigma: float,
        c_1: float, c_mu: float,
        mu_eff: float, device: str = 'cpu'):

        self.device = device

        self.num_params = num_params

        self.population_size = population_size
        self.parents_size = parents_size

        weights = math.log(self.parents_size + 0.5) - torch.log(torch.arange(1, self.parents_size + 1, device=device))
        self.weights = weights / weights.sum()

        self.sigma = sigma
        self.m = torch.zeros(num_params, device=device)
        self.C = torch.eye(num_params, device=device)

        self.p_sigma = torch.zeros(self.num_params, device=self.device)
        self.p_c = torch.zeros(self.num_params, device=self.device)

        self.expected_norm = self.num_params**(1/2) * (1 - 1.0 / (4 * self.num_params) + 1.0 / (21 * self.num_params**2))

        self.c_c = c_c
        self.c_1 = c_1
        self.c_mu = c_mu
        self.c_sigma = c_sigma
        self.d_sigma = d_sigma
        self.mu_eff = mu_eff

    def ask(self) -> torch.Tensor:
        C_stable = self.C + torch.eye(self.num_params, device=self.device) * 1e-6 
        cholesky_lower = torch.linalg.cholesky(C_stable * (self.sigma ** 2))

        dist = torch.distributions.MultivariateNormal(self.m, scale_tril=cholesky_lower)
        self.population = dist.sample((self.population_size,))

        return self.population

    def tell(self, fitness: List[float]):
        fitness_tensor = torch.tensor(fitness, device=self.device)

        sorted_indices = torch.argsort(fitness_tensor)
        best_indices = sorted_indices[:self.parents_size]
        best_individuals = self.population[best_indices] # [parents_size, num_params]

        # update mean and calculate mean update step
        old_m = self.m.clone() # [num_params]
        old_sigma = self.sigma

        self.m = torch.sum(self.weights.unsqueeze(1) * best_individuals, dim=0) # [,num_params]

        y_w = (self.m - old_m) / old_sigma # [num_params]

        # eigandecomposition to get C^(-1/2)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.C)

        inv_sqrt_D = torch.diag(1.0 / torch.sqrt(eigenvalues + 1e-8)) # [num_params, num_params]
        C_inv_sqrt = torch.mm(eigenvectors, torch.mm(inv_sqrt_D, eigenvectors.t())) # [num_params, num_params]

        # comulative step-size adaptation
        isotropic_y_w = torch.mv(C_inv_sqrt, y_w) # [num_params] here the y_w is reshaped back

        cs_norm = (self.c_sigma * (2 - self.c_sigma) * self.mu_eff) ** (1/2)

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + cs_norm * isotropic_y_w # [num_params]

        p_sigma_norm = torch.norm(self.p_sigma)
        step_adjustment = (self.c_sigma / self.d_sigma) * ((p_sigma_norm / self.expected_norm) - 1)
        self.sigma = self.sigma * torch.exp(step_adjustment) # [num_params]

        # covariance matrix adaptation
        cc_norm = (self.c_c * (2 - self.c_c) * self.mu_eff) ** (1/2)

        self.p_c = (1 - self.c_c) * self.p_c + cc_norm * y_w # [num_params]

        rank_1_update = torch.ger(self.p_c, self.p_c)

        Y = (best_individuals - old_m) / old_sigma # [mu, num_params]

        rank_mu_update = torch.mm(Y.t(), torch.mm(torch.diag(self.weights), Y))

        self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * rank_1_update + self.c_mu * rank_mu_update

           
