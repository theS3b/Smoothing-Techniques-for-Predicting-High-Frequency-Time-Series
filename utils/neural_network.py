import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.results import compute_rsquared
import seaborn as sns
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, 300),
            nn.ReLU(),
            nn.LayerNorm(300),  # Layer normalization (reduces overfitting)
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.LayerNorm(100),  # Layer normalization (reduces overfitting)
            nn.Linear(100, 1)   # Output layer
        )

    def forward(self, x):
        if len(x.shape) == 3:  # num_samples can be used to pass multiple samples at once (e.g. for specific loss)
            batch_size, num_samples, num_features = x.size()
            x = x.view(batch_size * num_samples, num_features)
            output = self.linear_relu_stack(x)
            output = output.view(batch_size, num_samples, -1)  # Reshape back
        elif len(x.shape) == 2:  # Validation: input shape (batch_size, 180)
            output = self.linear_relu_stack(x)
            # output = output.view(x.size(0), 1, -1)  # Reshape to (batch_size, 1, 1) for consistency
        else:
            raise ValueError("Input shape not supported!")
        return output
    
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

def train_nn(x_train, y_train, x_valid, y_valid, num_epochs=2000, learning_rate=1e-3, weight_decay=1e-3, custom_loss = MSELoss(), current_gdp_idx=None, seed = 42, verbose = True):
    set_seed(seed)

    device = get_device(verbose)

    num_features = x_train.shape[-1]
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
    mse_losses = []
    
    epoch_range = tqdm(range(num_epochs)) if verbose else range(num_epochs)
    for t in epoch_range:
        model.train()
        y_pred = model(x_train_t)
        loss_train = loss_fn(y_pred, y_train_t)

        model.eval()
        y_pred_valid = model(x_valid_t)
        loss_valid = loss_fn(y_pred_valid, y_valid_t)
        model.train()

        r_squared = compute_rsquared(y_valid, model(torch.tensor(x_valid, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten())

        valid_r_squared.append(r_squared)
        training_loss.append(loss_train.item())
        validation_loss.append(loss_valid.item())

        if len(y_pred.shape) == 3 and current_gdp_idx is not None and y_pred.shape[1] >= current_gdp_idx:
            mse_train = torch.linalg.norm(y_pred[:,current_gdp_idx,:] - y_train_t, ord=2).item() / y_train_t.size(0)
            mse_valid = torch.linalg.norm(y_pred_valid - y_valid_t, ord=2).item() / y_valid_t.size(0)
            mse_losses.append([mse_train, mse_valid])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
    model.eval()
    y_pred_valid = model(x_valid_t)

    v_loss = loss_fn(y_pred_valid, y_valid_t)
    validation_loss.append(v_loss.item())

    y_pred_train = model(x_train_t)
    t_loss = loss_fn(y_pred_train, y_train_t)
    training_loss.append(t_loss.item())

    r_squared = compute_rsquared(y_valid, model(torch.tensor(x_valid, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten())
    valid_r_squared.append(r_squared)

    if len(y_pred_train.shape) == 3 and current_gdp_idx is not None and y_pred_train.shape[1] >= current_gdp_idx:
        mse_train = torch.linalg.norm(y_pred_train[:,current_gdp_idx,:] - y_train_t, ord=2).item() / y_train_t.size(0)
        mse_valid = torch.linalg.norm(y_pred_valid - y_valid_t, ord=2).item() / y_valid_t.size(0)
        mse_losses.append([mse_train, mse_valid])

    if verbose:
        print(f"Final training loss: {t_loss.item()}")
        print(f"Final validation loss: {v_loss.item()}")
        print(f"Final validation R^2: {r_squared}")

        plt.figure(figsize=(10, 3))
        plt.plot(training_loss, label="Training loss")
        plt.plot(validation_loss, label="Validation loss")
        plt.plot(valid_r_squared, label="Validation $R^2$")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.legend()
        plt.show()

    
    return model, training_loss, validation_loss, valid_r_squared, mse_losses

def batch_data_by_country(df_data, df_hf_data, nb_prev, nb_after, other_cols_to_keep=None, verbose=True):
    pred_batches = []
    dates = []
    countries = []
    other_data_filtered = []
    all_dates = []

    for country in df_data['country'].unique():
        country_data = df_data[df_data['country'] == country]
        
        batch_dates = []
        for date_data in country_data.iterrows():
            cur_date = date_data[1]['date']

            # Append previous data
            batch = []
            for j in range(1, nb_prev+1):
                prev_tmp = df_hf_data[(df_hf_data['country'] == country) & (df_hf_data['date'] == cur_date - pd.DateOffset(months=j))]
                if prev_tmp is None or len(prev_tmp) == 0:
                    break

                batch.append(prev_tmp['data'].values[0].astype(np.float32))

            # Append current data
            batch.append(date_data[1]['data'])

            # Append next data
            for j in range(1, nb_after+1):
                next_tmp = df_hf_data[(df_hf_data['country'] == country) & (df_hf_data['date'] == cur_date + pd.DateOffset(months=j))]
                if next_tmp is None or len(next_tmp) == 0:
                    break

                batch.append(next_tmp['data'].values[0].astype(np.float32))

            if len(batch) != nb_prev + nb_after + 1:
                if verbose:
                    print("Skipping", country, cur_date)
                continue

            # Append batch
            pred_batches.append(batch)
            dates.append(cur_date)
            countries.append(country)
            batch_dates.append(cur_date)

            if other_cols_to_keep is not None:
                other_data_filtered.append(date_data[1][other_cols_to_keep].values)

        all_dates.append(batch_dates)

        
    return np.array(pred_batches), np.array(countries), np.array(dates), other_data_filtered if other_cols_to_keep is not None else None, all_dates