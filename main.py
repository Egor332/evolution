import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from models.abstraction import IBaseNeuralNetworkModel
from models.gradient_model_adamw import GradientModelAdamW


def prepare_wine_data(test_size: float = 0.2, random_state: int = 42, binary: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Download and prepare Wine Quality Dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    
    x = df.drop('quality', axis=1).values
    y_raw = df['quality'].values
    
    if binary:
        print("Using binary classification (quality >= 6)")
        y = (y_raw >= 6).astype(int)
    else:
        print("Using multiclass classification")
        y = y_raw - y_raw.min()
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    print(f"Data ready. Train size: {x_train_tensor.shape[0]}, test size: {x_test_tensor.shape[0]}")
    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor


def run_experiment(model: IBaseNeuralNetworkModel, x_train: torch.Tensor, y_train: torch.Tensor, iterations: int) -> List[float]:

    print(f"Start model learning for {iterations} iterations...")
    
    loss_history = model.train(x_train, y_train, iterations=iterations)
    
    print(f"Training finished. Final loss: {loss_history[-1]:.4f}")
    return loss_history


if __name__ == "__main__":
    # Set to True for binary classification (quality >= 6), False for multiclass
    use_binary_classification = False
    x_train, x_test, y_train, y_test = prepare_wine_data(binary=use_binary_classification)
    
    input_size = x_train.shape[1]
    output_size = len(torch.unique(y_train))
    hidden_sizes = [64, 32]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_adamw = GradientModelAdamW(
        input_size=input_size, 
        hidden_sizes=hidden_sizes, 
        output_size=output_size, 
        lr=0.001
    )
    
    history = run_experiment(
        model=model_adamw, 
        x_train=x_train, 
        y_train=y_train, 
        iterations=100
    )