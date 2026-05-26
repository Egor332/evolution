import numpy as np
import torch
from typing import List, Tuple, Optional

from models.abstraction import IBaseNeuralNetworkModel


class LShadeModel(IBaseNeuralNetworkModel):
    """
    L-SHADE: Success-history based adaptive DE with linear population size reduction.

    This implementation optimizes the flattened vector of all PyTorch parameters (weights + biases)
    by minimizing CrossEntropyLoss over the training set.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        *,
        population_size: int = 30,
        population_size_min: int = 4,
        memory_size: int = 10,
        pmin: float = 0.1,
        pmax: float = 0.2,
        cauchy_scale: float = 0.1,
        normal_scale: float = 0.1,
        bounds_scale: float = 5.0,
        seed: Optional[int] = None,
    ):
        super().__init__(input_size, hidden_sizes, output_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.population_size = int(population_size)
        self.population_size_min = int(population_size_min)
        self.memory_size = int(memory_size)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.cauchy_scale = float(cauchy_scale)
        self.normal_scale = float(normal_scale)
        self.bounds_scale = float(bounds_scale)
        self.seed = seed

        self._param_numel: Optional[int] = None
        self._param_shapes: Optional[List[Tuple[int, ...]]] = None

        # Filled in during first train() call.
        self._bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _ensure_vector_metadata(self) -> None:
        if self._param_numel is not None:
            return
        shapes: List[Tuple[int, ...]] = []
        numel_total = 0
        for p in self.model.parameters():
            shapes.append(tuple(p.shape))
            numel_total += p.numel()
        self._param_shapes = shapes
        self._param_numel = numel_total

    def _vectorize_parameters(self) -> np.ndarray:
        self._ensure_vector_metadata()
        vec_parts: List[np.ndarray] = []
        for p in self.model.parameters():
            vec_parts.append(p.detach().to("cpu", copy=True).numpy().ravel())
        return np.concatenate(vec_parts, axis=0).astype(np.float64, copy=False)

    def _set_parameters_from_vector(self, vector: np.ndarray) -> None:
        self._ensure_vector_metadata()
        assert self._param_numel is not None

        vector_t = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        if vector_t.numel() != self._param_numel:
            raise ValueError(
                f"Vector length mismatch: expected {self._param_numel}, got {vector_t.numel()}"
            )

        offset = 0
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                new_val = vector_t[offset : offset + numel].view_as(p)
                p.copy_(new_val)
                offset += numel

    def _compute_fitness(self, vector: np.ndarray, x_train: torch.Tensor, y_train: torch.Tensor) -> float:
        self._set_parameters_from_vector(vector)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_train)
            loss = self.criterion(logits, y_train)
        return float(loss.item())

    def _sample_cr(self, mu: float) -> float:
        # Normal sampling with truncation to [0, 1].
        cr = np.random.normal(mu, self.normal_scale)
        if cr < 0.0:
            cr = 0.0
        elif cr > 1.0:
            cr = 1.0
        return float(cr)

    def _sample_f(self, mu: float) -> float:
        # Cauchy sampling with resampling until in (0, 1]; this follows the "regenerate" idea
        # commonly used for JADE/SHADE implementations.
        for _ in range(1000):
            f = np.random.standard_cauchy() * self.cauchy_scale + mu
            if 0.0 < f <= 1.0:
                return float(f)
            # If it goes out of range, retry.
        # Fallback: clamp
        return float(np.clip(mu, 1e-12, 1.0))

    def _linear_population_size(self, gen: int, total_gens: int) -> int:
        # gen in [0, total_gens-1]
        np0 = self.population_size
        np_min = self.population_size_min
        if total_gens <= 1:
            return np_min
        t = gen / (total_gens - 1)
        # Round to keep integer population sizes.
        return int(np.round(np0 - (np0 - np_min) * t))

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        iterations: int = 100,
    ) -> list:
        self._ensure_vector_metadata()
        self.model.to(self.device)
        self.model.eval()

        if self.seed is not None:
            np.random.seed(self.seed)

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        # Bounds derived from initial parameters.
        init_vec = self._vectorize_parameters()
        abs_max = float(np.max(np.abs(init_vec)))
        abs_max = max(abs_max, 1e-3)
        lower = -self.bounds_scale * abs_max
        upper = self.bounds_scale * abs_max
        self._bounds = (
            np.full((self._param_numel,), lower, dtype=np.float64),
            np.full((self._param_numel,), upper, dtype=np.float64),
        )

        assert self._bounds is not None
        lb, ub = self._bounds

        # Initialize population.
        NP = self.population_size
        D = int(self._param_numel)
        pop = lb + np.random.rand(NP, D) * (ub - lb)
        fit = np.empty((NP,), dtype=np.float64)

        for i in range(NP):
            fit[i] = self._compute_fitness(pop[i], x_train, y_train)

        best_idx = int(np.argmin(fit))
        best_fitness = float(fit[best_idx])
        best_vector = pop[best_idx].copy()

        # Success-history memory.
        H = self.memory_size
        M_F = np.full((H,), 0.5, dtype=np.float64)
        M_CR = np.full((H,), 0.5, dtype=np.float64)
        k_mem = 0

        archive: List[np.ndarray] = []
        history: List[float] = []

        total_gens = int(iterations)
        if total_gens <= 0:
            self._set_parameters_from_vector(best_vector)
            return [best_fitness]

        # Precompute ordering each generation (cost: O(N log N), okay for small NP).
        for gen in range(total_gens):
            NP_target = self._linear_population_size(gen, total_gens)

            # Resize population if needed at the start of generation.
            if NP_target < pop.shape[0]:
                order = np.argsort(fit)[:NP_target]
                pop = pop[order]
                fit = fit[order]
                if len(archive) > NP_target:
                    keep = np.random.choice(len(archive), size=NP_target, replace=False)
                    archive = [archive[j] for j in keep]

            # Ensure best tracking.
            gen_best_idx = int(np.argmin(fit))
            if float(fit[gen_best_idx]) < best_fitness:
                best_fitness = float(fit[gen_best_idx])
                best_vector = pop[gen_best_idx].copy()

            # Sort indices for pbest.
            sorted_idx = np.argsort(fit)

            SF: List[float] = []
            SCR: List[float] = []
            dF: List[float] = []

            # Generate and evaluate trial vectors.
            for i in range(pop.shape[0]):
                x_i = pop[i]

                # Select memory index.
                r_idx = np.random.randint(H)
                mu_cr = float(M_CR[r_idx])
                mu_f = float(M_F[r_idx])

                cr_i = self._sample_cr(mu_cr)
                f_i = self._sample_f(mu_f)

                # Random p in [pmin, pmax]
                p_i = self.pmin + np.random.rand() * (self.pmax - self.pmin)
                p_best_count = int(max(2, np.ceil(p_i * pop.shape[0])))
                p_best_count = min(p_best_count, pop.shape[0])

                p_best_idx = sorted_idx[np.random.randint(0, p_best_count)]
                x_pbest = pop[p_best_idx]

                # r1 from current population, distinct from i.
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(pop.shape[0])
                x_r1 = pop[r1]

                # r2 from population union archive, distinct from i and r1 where possible.
                use_archive = len(archive) > 0
                if use_archive:
                    pool = np.vstack([pop, np.vstack(archive)])
                else:
                    pool = pop

                # draw r2 until it's not the target (and ideally not r1).
                # If pool is small, allow repeats with a bounded loop.
                r2_vec = None
                for _ in range(30):
                    cand = pool[np.random.randint(pool.shape[0])]
                    if not np.allclose(cand, x_i) and not np.allclose(cand, x_r1):
                        r2_vec = cand
                        break
                if r2_vec is None:
                    r2_vec = pool[np.random.randint(pool.shape[0])]

                # Mutation: current-to-pbest/1/bin (with archive).
                v = x_i + f_i * (x_pbest - x_i) + f_i * (x_r1 - r2_vec)
                # Binomial crossover.
                j_rand = np.random.randint(D)
                u = x_i.copy()
                rand_mask = np.random.rand(D) <= cr_i
                rand_mask[j_rand] = True
                u[rand_mask] = v[rand_mask]
                # Bounds handling (clip).
                u = np.clip(u, lb, ub)

                # Selection.
                f_u = self._compute_fitness(u, x_train, y_train)
                if f_u <= fit[i]:
                    # Archive stores replaced individuals.
                    archive.append(x_i.copy())
                    pop[i] = u
                    d_improve = float(fit[i] - f_u)
                    fit[i] = f_u

                    SF.append(f_i)
                    SCR.append(cr_i)
                    dF.append(d_improve)

            # Truncate archive to current pop size.
            if len(archive) > pop.shape[0]:
                keep = np.random.choice(len(archive), size=pop.shape[0], replace=False)
                archive = [archive[j] for j in keep]

            # Parameter adaptation with success-history.
            if len(SF) > 0 and float(np.sum(dF)) > 0.0:
                weights = np.asarray(dF, dtype=np.float64)
                weights_sum = float(np.sum(weights))
                if weights_sum <= 0:
                    weights = np.ones_like(weights) / len(weights)
                else:
                    weights = weights / weights_sum

                SF_arr = np.asarray(SF, dtype=np.float64)
                SCR_arr = np.asarray(SCR, dtype=np.float64)

                # Weighted Lehmer mean for F.
                MF_new = float(np.sum(weights * (SF_arr ** 2)) / np.sum(weights * SF_arr))
                # Weighted arithmetic mean for CR.
                MCR_new = float(np.sum(weights * SCR_arr))

                M_F[k_mem] = MF_new
                M_CR[k_mem] = MCR_new

            # Advance memory index.
            k_mem = (k_mem + 1) % H

            history.append(best_fitness)

        # Set best solution found.
        self._set_parameters_from_vector(best_vector)
        return history

