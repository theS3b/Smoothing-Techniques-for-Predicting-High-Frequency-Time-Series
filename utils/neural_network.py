import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from utils.results import compute_rsquared
from sklearn.metrics import mean_squared_error
import seaborn as sns

class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
# Function to set random seed
def set_seed(seed):
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # PyTorch for all GPUs (if multiple GPUs are used)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoids non-deterministic optimizations

def get_device(verbose = False):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if verbose:
        print(f"Using {device} device")
    return device
    
def train_nn(x_train, y_train, x_valid, y_valid, num_epochs=2000, learning_rate=1e-3, weight_decay=1e-3, custom_loss = MSELoss(), seed = 42, verbose = True):
    set_seed(seed)

    device = get_device(verbose)

    num_features = x_train.shape[1]
    model = NeuralNetwork(num_features=num_features).to(device)
    loss_fn = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)
    x_valid_t = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_valid_t = torch.tensor(y_valid, dtype=torch.float32).to(device).unsqueeze(1)

    training_loss = []
    validation_loss = []
    valid_r_squared = []
    
    epoch_range = tqdm(range(num_epochs)) if verbose else range(num_epochs)
    for t in epoch_range:
        model.train()
        y_pred = model(x_train_t)
        loss_train = loss_fn(y_pred, y_train_t)

        model.eval()
        loss_valid = loss_fn(model(x_valid_t), y_valid_t)
        model.train()

        r_squared = compute_rsquared(y_valid, model(torch.tensor(x_valid, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten())

        valid_r_squared.append(r_squared)
        training_loss.append(loss_train.item())
        validation_loss.append(loss_valid.item())

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
    model.eval()
    y_pred = model(x_valid_t)

    v_loss = loss_fn(y_pred, y_valid_t)
    validation_loss.append(v_loss.item())

    t_loss = loss_fn(model(x_train_t), y_train_t)
    training_loss.append(t_loss.item())

    r_squared = compute_rsquared(y_valid, model(torch.tensor(x_valid, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten())
    valid_r_squared.append(r_squared)

    if verbose:
        print(f"Final training loss: {t_loss.item()}")
        print(f"Final validation loss: {v_loss.item()}")
        print(f"Final validation R^2: {r_squared}")

        plt.figure(figsize=(10, 3))
        plt.plot(training_loss, label="Training loss")
        plt.plot(validation_loss, label="Validation loss")
        plt.plot(valid_r_squared, label="Validation R^2")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.legend()
        plt.show()

    
    return model, training_loss, validation_loss, valid_r_squared

def summarize_results(y_valid, y_pred_mean, rsquared_ensemble):
    # Plot the r squared
    ensemble_r2 = compute_rsquared(y_valid, y_pred_mean)
    ensemble_mse = mean_squared_error(y_valid, y_pred_mean)
    ensemble_mape = np.mean(np.abs((y_valid - y_pred_mean) / y_valid)) * 100
    print(f"Ensemble R2: {ensemble_r2}")
    print(f"Ensemble MSE: {ensemble_mse}")
    print(f"Ensemble MAPE: {ensemble_mape}")

    plt.figure(figsize=(10, 3))
    sns.histplot(rsquared_ensemble, bins=30, kde=True)
    plt.xlabel("R squared")
    plt.ylabel("Density")
    plt.title("Distribution of R squared values")
    plt.show()