import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
import pywt
from scipy.interpolate import UnivariateSpline
import seaborn as sns

def compute_rsquared(y_true, y_pred):
    """
    Compute the R-squared value of the predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean = np.mean(y_true)
    ss_total = np.sum((y_true - mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_total


def bootstrap_ensemble(
    X_valid,
    y_valid,
    train_fn,  # Callable for training the model, can be a lambda
    n_ensembling=100,
    seed=42,
    device=torch.device("cpu"),
    verbose=False,
    metrics=None,  # Dictionary of metric functions
):
    """
    Performs bootstrapping ensembling by training multiple models with different seeds.

    Parameters:
    - X_valid (np.ndarray): Validation features
    - y_valid (np.ndarray): Validation targets
    - train_fn (Callable): A function that trains a model and returns it
    - n_ensembling (int): Number of models to train
    - seed (int): Seed for reproducibility
    - verbose (bool): Whether to display progress bars
    - metrics (dict): Dictionary of metric functions

    Returns:
    dict: A dictionary containing the ensemble models, their metrics, aggregated predictions,
          the best model based on R-squared, and optionally the number of unique countries.
    """
    
    # Set default metrics if none are provided
    if metrics is None:
        metrics = {
            'mse': mean_squared_error,
            'rsquared': r2_score
        }

    # Initialize containers for models and metrics
    bootstrap_models = []
    mse_ensemble = np.zeros(n_ensembling)
    rsquared_ensemble = np.zeros(n_ensembling)

    for i in tqdm(range(n_ensembling), desc="Bootstrapping Ensembling"):
        current_seed = seed + i

        # Train the model using the provided train_fn
        result = train_fn(current_seed)

        single_model = result[0]  # Assuming the first element is the trained model
        
        # Generate predictions on the validation set
        y_pred = single_model(
            torch.tensor(X_valid, dtype=torch.float32).to(device)
        ).cpu().detach().numpy().flatten()

        # Calculate metrics
        mse = metrics['mse'](y_valid, y_pred)
        rsquared = metrics['rsquared'](y_valid, y_pred)

        # Store the model and its metrics
        bootstrap_models.append(single_model)
        mse_ensemble[i] = mse
        rsquared_ensemble[i] = rsquared

    # Aggregate predictions from all models
    y_pred_aggregate = np.zeros((X_valid.shape[0], n_ensembling))
    for i, model in enumerate(bootstrap_models):
        y_pred_aggregate[:, i] = model(
            torch.tensor(X_valid, dtype=torch.float32).to(device)
        ).cpu().detach().numpy().flatten()

    # Compute aggregated statistics
    y_pred_mean = np.mean(y_pred_aggregate, axis=1)
    y_pred_std = np.std(y_pred_aggregate, axis=1)
    y_pred_median = np.median(y_pred_aggregate, axis=1)

    # Identify the best model based on R-squared
    best_model_idx = np.argmax(rsquared_ensemble)
    best_model = bootstrap_models[best_model_idx]
    best_rsquared = rsquared_ensemble[best_model_idx]
    y_pred_best = best_model(
        torch.tensor(X_valid, dtype=torch.float32).to(device)
    ).cpu().detach().numpy().flatten()

    return {
        'bootstrap_models': bootstrap_models,
        'mse_ensemble': mse_ensemble,
        'rsquared_ensemble': rsquared_ensemble,
        'y_pred_mean': y_pred_mean,
        'y_pred_std': y_pred_std,
        'y_pred_median': y_pred_median,
        'best_model': best_model,
        'best_rsquared': best_rsquared,
        'y_pred_best': y_pred_best
    }


def plot_predictions_by_country(
    selected_country,
    country_valid,
    y_valid,
    y_pred_mean,
    y_pred_median=None,
    y_pred_best=None,
    y_pred_std=None,
    figsize=(15, 5),
    title_prefix="",
    show_mean=True,
    show_median=True,
    show_best=True,
    fill_confidence=True,
    alpha_fill=0.05,
    alpha_fill_confidence=0.2,
    std_multipliers=(3, 1.96),
    
):
    """
    Plots the true and predicted GDP values for a selected country over time, including confidence intervals.
    
    Parameters:
    - selected_country (str): The country to plot.
    - country_valid (np.ndarray): An array of country names.
    - y_valid (np.ndarray): The true GDP values.
    - y_pred_mean (np.ndarray): The predicted GDP values (mean).
    - y_pred_median (np.ndarray): The predicted GDP values (median).
    - y_pred_best (np.ndarray): The predicted GDP values (best model).
    - y_pred_std (np.ndarray): The standard deviation of the predicted GDP values.
    - figsize (tuple): The figure size.
    - title_prefix (str): A prefix to add to the plot title.
    - show_mean (bool): Whether to plot the predicted mean values.
    - show_median (bool): Whether to plot the predicted median values.
    - show_best (bool): Whether to plot the predicted best values.
    - fill_confidence (bool): Whether to fill the confidence intervals.
    - labels (dict): A dictionary of labels for the plot.
    - colors (dict): A dictionary of colors for the plot.
    - alpha_fill (float): The transparency of the confidence interval fill.
    - alpha_fill_confidence (float): The transparency of the confidence interval fill for the second interval.
    - std_multipliers (tuple): Multipliers for the standard deviation to determine the confidence intervals.

    
    
    Returns:
    - None: Displays the plot.
    """
    
    # Set default labels if none are provided
    labels = {
        'true': "True",
        'pred_mean': "Predicted (Mean)",
        'pred_median': "Predicted (Median)",
        'pred_best': "Predicted (Best)"
    }
    
    # Set default colors if none are provided
    colors = {
        'true': 'blue',
        'pred_mean': 'orange',
        'pred_median': 'green',
        'pred_best': 'purple',
        'confidence_interval': 'red'
    }

    xlabel="Date",
    ylabel="GDP"

    # Filter data for the selected country
    mask = (country_valid == selected_country)
    
    if not np.any(mask):
        raise ValueError(f"No data found for the selected country: {selected_country}")
    
    y_true = y_valid[mask]

    show_mean = show_mean and y_pred_mean is not None
    show_median = show_median and y_pred_median is not None
    show_best = show_best and y_pred_best is not None
    fill_confidence = fill_confidence and y_pred_std is not None

    if show_mean:
        y_mean = y_pred_mean[mask]
    if show_median:
        y_median = y_pred_median[mask]
    if show_best:
        y_best = y_pred_best[mask]
    if fill_confidence:
        y_std = y_pred_std[mask]
    
    # Define the x-axis as a range of dates or indices
    x = np.arange(len(y_true))
    
    plt.figure(figsize=figsize)
    
    # Plot true values
    plt.plot(x, y_true, label=labels.get('true', "True"), color=colors.get('true', 'blue'))
    
    # Plot predicted mean
    if show_mean:
        plt.plot(x, y_mean, label=labels.get('pred_mean', "Predicted (Mean)"), color=colors.get('pred_mean', 'orange'))
    
    # Plot predicted median
    if show_median:
        plt.plot(x, y_median, label=labels.get('pred_median', "Predicted (Median)"), color=colors.get('pred_median', 'green'))
    
    # Plot predicted best
    if show_best:
        plt.plot(x, y_best, label=labels.get('pred_best', "Predicted (Best)"), color=colors.get('pred_best', 'purple'))
    
    # Plot confidence intervals
    if fill_confidence:
        multiplier1, multiplier2 = std_multipliers
        
        # First confidence interval (e.g., ±3σ)
        plt.fill_between(
            x,
            y_mean - multiplier1 * y_std,
            y_mean + multiplier1 * y_std,
            color=colors.get('confidence_interval', 'red'),
            alpha=alpha_fill,
            label=f'Confidence Interval (±{multiplier1}σ)'
        )
        
        # Second confidence interval (e.g., ±1.96σ for ~95% confidence)
        plt.fill_between(
            x,
            y_mean - multiplier2 * y_std,
            y_mean + multiplier2 * y_std,
            color=colors.get('confidence_interval', 'red'),
            alpha=alpha_fill_confidence,
            label=f'Confidence Interval (±{multiplier2}σ)'
        )
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}{selected_country}" if title_prefix else f"{selected_country}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example Usage with ipywidgets.interact
def interactive_plot_predictions(
    country_valid,
    y_valid,
    y_pred_mean,
    y_pred_median,
    y_pred_best,
    y_pred_std,
    title_prefix="GDP Prediction: "
):
    """
    Creates an interactive widget to plot predictions by country.
    
    Parameters:
    - All parameters are passed to the plot_predictions_by_country function.
    
    Returns:
    - An interactive widget.
    """
    unique_countries = np.unique(country_valid)
    
    interact(
        lambda selected_country: plot_predictions_by_country(
            selected_country=selected_country,
            country_valid=country_valid,
            y_valid=y_valid,
            y_pred_mean=y_pred_mean,
            y_pred_median=y_pred_median,
            y_pred_best=y_pred_best,
            y_pred_std=y_pred_std,
            title_prefix=title_prefix
        ),
        selected_country=unique_countries
    )


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

def std_first_derivative(series):
    """Standard deviation of the first derivative (finite differences)."""
    diff = np.diff(series)
    return np.std(diff) / series.shape[0]

def mean_abs_first_difference(series):
    """Mean absolute first difference."""
    diff = np.diff(series)
    return np.mean(np.abs(diff)) / series.shape[0]

def std_second_derivative(series):
    """Variance or standard deviation of the second derivative."""
    second_diff = np.diff(series, n=2)
    return np.std(second_diff) / series.shape[0]

def total_variation(series):
    """Total variation."""
    diff = np.diff(series)
    return np.sum(np.abs(diff)) / series.shape[0]

def high_frequency_energy(series, cutoff=0.1):
    """Fourier-based high-frequency energy."""
    fft = np.fft.fft(series)
    freqs = np.fft.fftfreq(len(series))
    
    reconstructed = np.copy(fft)
    reconstructed[np.abs(freqs) > cutoff] = 0

    filtered = np.fft.ifft(reconstructed)
    return np.linalg.norm(filtered - series) ** 2 / series.shape[0]

def wavelet_smoothness(series, wavelet='db1'):
    """Wavelet-based smoothness measure."""
    coeffs = pywt.wavedec(series, wavelet)
    detail_coeffs = coeffs[1:]  # Skip approximation coefficients
    return np.sum([np.sum(c**2) for c in detail_coeffs]) / series.shape[0]

def holder_exponent(series):
    """Estimate the Hölder exponent."""
    diff = np.abs(np.diff(series))
    return -np.log(np.mean(diff)) / np.log(len(series)) / series.shape[0]  # Simplified estimate

def sobolev_norm(series):
    """Sobolev norm (L2 norm of first derivative)."""
    diff = np.diff(series)
    return np.sqrt(np.sum(diff**2)) / series.shape[0]

def integrated_abs_curvature(series):
    """Integrated absolute curvature."""
    second_diff = np.abs(np.diff(series, n=2))
    return np.sum(second_diff) / series.shape[0]

def spline_roughness(series):
    """Spline-based roughness penalty."""
    x = np.arange(len(series))
    spline = UnivariateSpline(x, series, s=0)
    second_derivative = spline.derivative(n=2)
    return np.sum(second_derivative(x)**2) / series.shape[0]

all_smoothness_metrics = [
    std_first_derivative,
    mean_abs_first_difference,
    std_second_derivative,
    total_variation,
    high_frequency_energy,
    wavelet_smoothness,
    holder_exponent,
    sobolev_norm,
    integrated_abs_curvature,
    spline_roughness
]