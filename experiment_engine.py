import os
import random
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.abstraction import IBaseNeuralNetworkModel


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_type: str  # 'binary' or 'multiclass'
    test_size: float = 0.2
    iterations: int = 100
    seed: int = 42


class ExperimentEngine:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        
        # Define output directories
        self.dirs = {
            "images": os.path.join("experiments_result", "images"),
            "statistics": os.path.join("experiments_result", "statistics"),
            "models": "trained_models"
        }
        
        # Create directories if they don't exist
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        self.set_seed(self.config.seed)
        
    def set_seed(self, seed: int):
        """Ensures 100% repeatability for the experiment."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def evaluate(self, model: IBaseNeuralNetworkModel, x_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """Calculates evaluation metrics on the test dataset."""
        outputs = model.predict(x_test)
        
        # Convert predictions to class indices if multiclass, or binary indices
        _, predicted = torch.max(outputs.data, 1)
        
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        # Select average method based on dataset classification
        avg_method = 'binary' if self.config.dataset_type == 'binary' else 'weighted'
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg_method, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg_method, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=avg_method, zero_division=0)
        }
        
        return metrics

    def run_experiment(self, 
                       model_factories: Dict[str, Callable[[], IBaseNeuralNetworkModel]], 
                       data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Runs the experiment for each provided model factory.
        Model factories allow creating models *after* setting the seed.
        """
        x_train, x_test, y_train, y_test = data
        
        print(f"Starting Experiment: {self.config.experiment_name}")
        print(f"Seed: {self.config.seed} | Type: {self.config.dataset_type}")
        
        for model_name, model_factory in model_factories.items():
            print(f"\n--- Running Model: {model_name} ---")
            
            # We explicitly reset the seed before initializing and training each model
            # to ensure true isolation and repeatability per model run if they were run individually
            self.set_seed(self.config.seed)
            
            model = model_factory()
            
            # Train the model
            loss_history = model.train(x_train, y_train, iterations=self.config.iterations)
            
            # Evaluate the model
            metrics = self.evaluate(model, x_test, y_test)
            
            # Save the model
            model_filepath = os.path.join(self.dirs["models"], f"{self.config.experiment_name}_{model_name}.pth")
            model.save(model_filepath)
            
            # Store results
            self.results[model_name] = {
                "loss_history": loss_history,
                "metrics": metrics,
                "model_filepath": model_filepath
            }
            
            print(f"[{model_name}] Training finished. Final loss: {loss_history[-1]:.4f}")
            for m_name, m_val in metrics.items():
                print(f"[{model_name}] {m_name.capitalize()}: {m_val:.4f}")
                
        self._save_statistics()
        self._plot_results()
        
    def _save_statistics(self):
        """Saves evaluation metrics to a JSON file."""
        stats = {}
        for model_name, result in self.results.items():
            stats[model_name] = result["metrics"]
            
        stats_file = os.path.join(self.dirs["statistics"], f"{self.config.experiment_name}_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"\nStatistics saved to {stats_file}")

    def _plot_results(self):
        """Plots training loss history and metrics comparison bar charts."""
        # 1. Plot Loss History
        plt.figure(figsize=(10, 6))
        for model_name, result in self.results.items():
            plt.plot(result["loss_history"], label=model_name)
            
        plt.title(f"Training Loss - {self.config.experiment_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        loss_img = os.path.join(self.dirs["images"], f"{self.config.experiment_name}_loss.png")
        plt.savefig(loss_img)
        plt.close()
        
        # 2. Plot Metrics Bar Chart
        if not self.results:
            return
            
        metrics_names = list(list(self.results.values())[0]["metrics"].keys())
        model_names = list(self.results.keys())
        
        x = np.arange(len(metrics_names))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, model_name in enumerate(model_names):
            metrics_vals = [self.results[model_name]["metrics"][m] for m in metrics_names]
            ax.bar(x + i*width - 0.4 + width/2, metrics_vals, width, label=model_name)
            
        ax.set_ylabel('Scores')
        ax.set_title(f"Test Metrics Comparison - {self.config.experiment_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').capitalize() for m in metrics_names])
        ax.legend()
        plt.ylim([0, 1.1])
        
        metrics_img = os.path.join(self.dirs["images"], f"{self.config.experiment_name}_metrics.png")
        plt.savefig(metrics_img)
        plt.close()
        
        print(f"Plots saved to {loss_img} and {metrics_img}")
