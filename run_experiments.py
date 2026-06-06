import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.cma_es_model import CMAESModel
from models.lshade_model import LShadeModel
from models.gradient_model_adamw import GradientModelAdamW
from experiment_engine import ExperimentEngine, ExperimentConfig

# --- Data Preparation ---
def get_dataset(seed: int = 42):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data_file = "winequality-red.csv"
    if not os.path.exists(data_file):
        df = pd.read_csv(url, sep=";")
        df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)
        
    X = df.drop("quality", axis=1).values
    y_multi = df["quality"].values
    
    # Remap multiclass labels to 0-indexed (classes are 3,4,5,6,7,8)
    unique_classes = np.sort(np.unique(y_multi))
    class_map = {val: idx for idx, val in enumerate(unique_classes)}
    y_multi_mapped = np.array([class_map[val] for val in y_multi])
    
    # Binary labels
    y_binary = (df["quality"] >= 6).astype(int).values

    # Scale X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split indices
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=y_multi_mapped)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=y_multi_mapped[temp_idx])

    # Save splits
    np.savez("dataset_split.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, class_map=class_map)

    # Convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    
    # Create datasets
    def make_data(y):
        y_t = torch.tensor(y, dtype=torch.long)
        return (X_t[train_idx], X_t[val_idx], X_t[test_idx], 
                y_t[train_idx], y_t[val_idx], y_t[test_idx])

    return make_data(y_binary), make_data(y_multi_mapped), unique_classes

# --- Topology Generator ---
def find_hidden_sizes(target_params, input_dim=11, output_dim=2):
    # We will use two hidden layers [h, h]
    # params = (input_dim + 1)*h + (h + 1)*h + (h + 1)*output_dim
    # h^2 + (input_dim + output_dim + 2)h + output_dim - target_params = 0
    b = input_dim + output_dim + 2
    c = output_dim - target_params
    h = (-b + np.sqrt(b**2 - 4*c)) / 2
    h = int(round(h))
    return [h, h]

def get_param_count(input_dim, hidden_sizes, output_dim):
    cnt = 0
    curr = input_dim
    for hs in hidden_sizes:
        cnt += (curr + 1) * hs
        curr = hs
    cnt += (curr + 1) * output_dim
    return cnt

# --- Model Factories ---
def create_cma_es(input_dim, hidden_sizes, output_dim):
    D = get_param_count(input_dim, hidden_sizes, output_dim)
    pop_size = max(4, int(4 + 3 * np.log(D)))
    return CMAESModel(input_dim, hidden_sizes, output_dim, sigma0=0.2, population_size=pop_size)

def create_lshade(input_dim, hidden_sizes, output_dim):
    return LShadeModel(
        input_dim, hidden_sizes, output_dim,
        population_size=300, population_size_min=4, memory_size=6,
        pmin=0.1, pmax=0.2, cauchy_scale=0.1, normal_scale=0.1
    )

def create_adamw(input_dim, hidden_sizes, output_dim):
    return GradientModelAdamW(input_dim, hidden_sizes, output_dim, lr=0.001)

# --- Experiments ---
def main():
    data_binary, data_multi, unique_classes = get_dataset(seed=42)
    num_classes = len(unique_classes)
    input_dim = 11

    # --- Experiment 1: Reality Check ---
    print("\n" + "="*50)
    print("EXPERIMENT 1: Reality Check")
    print("="*50)
    hs_500 = find_hidden_sizes(500, input_dim, 2)
    print(f"Chosen hidden sizes for ~500 params: {hs_500} (Actual: {get_param_count(input_dim, hs_500, 2)})")
    
    # 3 seeds
    e1_results = {"cmaes": {"prec": [], "rec": []}, "lshade": {"prec": [], "rec": []}, "adamw": {"prec": [], "rec": []}}
    for seed in [42, 43, 44]:
        config = ExperimentConfig(
            experiment_name=f"Exp1_RealityCheck_seed_{seed}",
            dataset_type="binary",
            max_nfe=15000,
            patience=800,
            seed=seed
        )
        engine = ExperimentEngine(config)
        factories = {
            "cmaes": lambda: create_cma_es(input_dim, hs_500, 2),
            "lshade": lambda: create_lshade(input_dim, hs_500, 2),
            "adamw": lambda: create_adamw(input_dim, hs_500, 2)
        }
        engine.run_experiment(factories, data_binary)
        for m in ["cmaes", "lshade", "adamw"]:
            e1_results[m]["prec"].append(engine.results[m]["metrics"]["precision"])
            e1_results[m]["rec"].append(engine.results[m]["metrics"]["recall"])
            
    # Plot Mean P/R for Exp1
    plt.figure(figsize=(8, 5))
    models = list(e1_results.keys())
    prec_means = [np.mean(e1_results[m]["prec"]) for m in models]
    rec_means = [np.mean(e1_results[m]["rec"]) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, prec_means, width, label='Precision')
    plt.bar(x + width/2, rec_means, width, label='Recall')
    plt.ylabel('Scores')
    plt.title('Experiment 1: Mean Precision and Recall over 3 seeds')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(os.path.join("experiments_result", "images", "Exp1_Summary_Metrics.png"))
    plt.close()

    # --- Experiment 2: Parameters Amount ---
    print("\n" + "="*50)
    print("EXPERIMENT 2: Parameters Amount")
    print("="*50)
    sizes = [100, 300, 500, 1000, 2000]
    e2_results = {m: {"prec": [], "rec": [], "time": []} for m in ["cmaes", "lshade", "adamw"]}
    import time
    
    for size in sizes:
        hs = find_hidden_sizes(size, input_dim, 2)
        print(f"\nEvaluating size {size} -> hidden layers {hs} (Actual params: {get_param_count(input_dim, hs, 2)})")
        
        # We store temp results for 5 seeds for this size
        temp_metrics = {m: {"prec": [], "rec": [], "time": []} for m in ["cmaes", "lshade", "adamw"]}
        
        for seed in range(100, 105):  # 5 seeds
            config = ExperimentConfig(
                experiment_name=f"Exp2_Size_{size}_seed_{seed}",
                dataset_type="binary",
                max_nfe=15000,
                patience=800,
                seed=seed
            )
            engine = ExperimentEngine(config)
            factories = {
                "cmaes": lambda: create_cma_es(input_dim, hs, 2),
                "lshade": lambda: create_lshade(input_dim, hs, 2),
                "adamw": lambda: create_adamw(input_dim, hs, 2)
            }
            
            for m_name, m_fact in factories.items():
                start_t = time.time()
                engine.run_experiment({m_name: m_fact}, data_binary)
                end_t = time.time()
                temp_metrics[m_name]["time"].append(end_t - start_t)
                temp_metrics[m_name]["prec"].append(engine.results[m_name]["metrics"]["precision"])
                temp_metrics[m_name]["rec"].append(engine.results[m_name]["metrics"]["recall"])
                
        for m in ["cmaes", "lshade", "adamw"]:
            e2_results[m]["prec"].append(np.mean(temp_metrics[m]["prec"]))
            e2_results[m]["rec"].append(np.mean(temp_metrics[m]["rec"]))
            e2_results[m]["time"].append(np.mean(temp_metrics[m]["time"]))

    # Plot Exp 2 results - Save as separate files
    for metric, title, filename in [
        ("prec", "Precision vs Network Size", "Exp2_Precision_vs_Size.png"),
        ("rec", "Recall vs Network Size", "Exp2_Recall_vs_Size.png"),
        ("time", "Computation Time (s) vs Network Size", "Exp2_Time_vs_Size.png")
    ]:
        plt.figure(figsize=(8, 5))
        for m in ["cmaes", "lshade", "adamw"]:
            plt.plot(sizes, e2_results[m][metric], marker='o', label=m)
        plt.title(title)
        plt.xlabel('Network Size')
        plt.ylabel(metric.capitalize() if metric != "time" else "Time (seconds)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("experiments_result", "images", filename))
        plt.close()

    # --- Pick Best Size for Exp 3 ---
    # Based on AdamW F1 / Precision, let's just pick 1000 for automation, or the best one
    # Simple logic: pick size with best sum of precision+recall for AdamW
    best_idx = np.argmax(np.array(e2_results["adamw"]["prec"]) + np.array(e2_results["adamw"]["rec"]))
    best_size = sizes[best_idx]
    best_hs = find_hidden_sizes(best_size, input_dim, num_classes)
    
    print("\n" + "="*50)
    print(f"EXPERIMENT 3: Multiclass Prediction (Best Size: {best_size})")
    print("="*50)
    
    config3 = ExperimentConfig(
        experiment_name=f"Exp3_Multiclass_size_{best_size}",
        dataset_type="multiclass",
        max_nfe=15000,
        patience=800,
        seed=200
    )
    engine3 = ExperimentEngine(config3)
    factories3 = {
        "cmaes": lambda: create_cma_es(input_dim, best_hs, num_classes),
        "lshade": lambda: create_lshade(input_dim, best_hs, num_classes),
        "adamw": lambda: create_adamw(input_dim, best_hs, num_classes)
    }
    engine3.run_experiment(factories3, data_multi)
    
    # Evaluate confusion matrix and F1 scores for Exp 3
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
    _, _, x_test, _, _, y_test = data_multi
    
    for m_name in ["cmaes", "lshade", "adamw"]:
        model = factories3[m_name]()
        model.load(engine3.results[m_name]["model_filepath"])
        outputs = model.predict(x_test)
        _, preds = torch.max(outputs, 1)
        
        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
        disp.plot()
        plt.title(f"Confusion Matrix - {m_name.upper()} (Multiclass)")
        plt.tight_layout()
        plt.savefig(os.path.join("experiments_result", "images", f"Exp3_ConfMatrix_{m_name}.png"))
        plt.close()
        
        # 2. F1 Score per Class & Macro F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Print results to console
        print(f"\n[{m_name.upper()}] Multiclass Metrics:")
        print(f"  Macro F1 Score: {macro_f1:.4f}")
        for idx, cls in enumerate(unique_classes):
            print(f"  Class {cls} F1 Score: {f1_per_class[idx]:.4f}")
            
        # Plot F1 Scores
        plt.figure(figsize=(8, 5))
        x_indices = np.arange(len(unique_classes))
        bars = plt.bar(x_indices, f1_per_class, color='skyblue', edgecolor='black', alpha=0.8)
        
        plt.title(f"F1 Score per Class - {m_name.upper()} (Multiclass)")
        plt.xlabel("Class (Wine Quality)")
        plt.ylabel("F1 Score")
        plt.xticks(x_indices, unique_classes)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add F1 score values on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
            
        # Write Macro F1 score inside the plot
        plt.text(0.95, 0.95, f"Macro F1: {macro_f1:.4f}", 
                 transform=plt.gca().transAxes, 
                 fontsize=12, fontweight='bold',
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join("experiments_result", "images", f"Exp3_F1_Scores_{m_name}.png"))
        plt.close()

    # --- Experiment 4: Statistical Stability ---
    print("\n" + "="*50)
    print("EXPERIMENT 4: Statistical Stability")
    print("="*50)
    
    e4_hs = find_hidden_sizes(best_size, input_dim, 2)
    e4_results = {m: {"prec": [], "rec": []} for m in ["cmaes", "lshade", "adamw"]}
    
    for seed in range(300, 315): # 15 seeds
        config4 = ExperimentConfig(
            experiment_name=f"Exp4_Stability_seed_{seed}",
            dataset_type="binary",
            max_nfe=15000,
            patience=800,
            seed=seed
        )
        engine4 = ExperimentEngine(config4)
        factories4 = {
            "cmaes": lambda: create_cma_es(input_dim, e4_hs, 2),
            "lshade": lambda: create_lshade(input_dim, e4_hs, 2),
            "adamw": lambda: create_adamw(input_dim, e4_hs, 2)
        }
        engine4.run_experiment(factories4, data_binary)
        
        for m in ["cmaes", "lshade", "adamw"]:
            e4_results[m]["prec"].append(engine4.results[m]["metrics"]["precision"])
            e4_results[m]["rec"].append(engine4.results[m]["metrics"]["recall"])

    # Boxplots - Save as separate files
    model_labels = ["CMA-ES", "L-SHADE", "AdamW"]
    
    # 1. Precision Stability Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([e4_results["cmaes"]["prec"], e4_results["lshade"]["prec"], e4_results["adamw"]["prec"]], labels=model_labels)
    plt.title("Precision Stability (15 runs)")
    plt.ylabel("Precision")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("experiments_result", "images", "Exp4_Precision_Stability.png"))
    plt.close()
    
    # 2. Recall Stability Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([e4_results["cmaes"]["rec"], e4_results["lshade"]["rec"], e4_results["adamw"]["rec"]], labels=model_labels)
    plt.title("Recall Stability (15 runs)")
    plt.ylabel("Recall")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("experiments_result", "images", "Exp4_Recall_Stability.png"))
    plt.close()

if __name__ == "__main__":
    main()
