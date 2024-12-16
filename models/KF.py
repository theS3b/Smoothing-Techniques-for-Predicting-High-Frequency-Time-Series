import numpy as np
from filterpy.kalman import KalmanFilter
import torch
import pandas as pd
from utils.neural_network import get_device
from utils.results import compute_rsquared, measure_smoothness_with_df

class KF:
    """
    Kalman filter model to smooth the predictions of a neural network, with a constant acceleration model as the theoretical model.

    Data is expected to be country-specific, and the model will create a filter for each country.
    """
    def __init__(self, nb_init_samples=5):
        """
        Initialize the Kalman filter model
        
        nb_init_samples: int, the number of initial samples to use to estimate the initial state (position, velocity, acceleration) of the Kalman filter.
        """
        self.nb_init_samples = nb_init_samples

        # KFs for each country
        self.kfs = {}

    def fit(self, y_pred, y_true, countries, accel_var=1e-4):
        """
        Fit the Kalman filters to the data
        
        y_pred: np.array of shape (n_samples, ) containing the predictions
        y_true: np.array of shape (n_samples, ) containing the true values
        countries: np.array of shape (n_samples, ) containing the country of each sample
        theoretical_noise_var: float, the theoretical model noise variance, less variance means smoother predictions. Defaults to 1e-4.
        """
        dt = 1 # Interval between observations
        nn_noise_var = np.var(y_pred - y_true)

        self.kfs = {}
        for country in np.unique(countries):
            country_data = y_pred[countries == country]

            # As initial values, we use the mean of (position, velocity, acceleration) over the first n samples
            first_n_samples = country_data[:self.nb_init_samples]
            first_n_velocities = [country_data[i + 1] - country_data[i] for i in range(self.nb_init_samples)]
            first_n_accelerations = [country_data[i + 2] - 2 * country_data[i + 1] + country_data[i] for i in range(self.nb_init_samples)]

            initial_gdp, initial_gdp_var = np.mean(first_n_samples), np.var(first_n_samples)
            initial_velocity, initial_velocity_var = np.mean(first_n_velocities), np.var(first_n_velocities)
            initial_acceleration, initial_acceleration_var = np.mean(first_n_accelerations), np.var(first_n_accelerations)

            self.kfs[country] = KalmanFilter(dim_x=3, dim_z=1)
            
            self.kfs[country].x = np.array([initial_gdp, initial_velocity, initial_acceleration])   # Initial state
            self.kfs[country].F = np.array([[1, dt, 0.5 * dt**2],       # position = position + velocity*dt + 0.5*acceleration*dt^2
                                            [0, 1, dt],                 # velocity = velocity + acceleration*dt
                                            [0, 0, 1]])                 # State transition matrix (assuming constant acceleration)
            self.kfs[country].Q = accel_var * np.eye(3)         # Prediction (constant acceleration model) noise covariance
            self.kfs[country].H = np.array([[1, 0, 0]])         # Measurement matrix, only position is observed
            self.kfs[country].R = nn_noise_var                  # Measurement noise covariance, the noise in the neural network predictions
            self.kfs[country].P = np.diag([                     # State covariance
                initial_gdp_var,
                initial_velocity_var,
                initial_acceleration_var
            ])

    def predict_update(self, y, countries):
        """
        Predict the next value and update the Kalman filters

        y: np.array of shape (n_samples, ) containing the predictions
        countries: np.array of shape (n_samples, ) containing the country of each sample

        Returns:
        np.array of shape (n_samples, ) containing the predictions
        """
        kf_predictions = []
        for prediction, country in zip(y, countries):
            if country not in self.kfs:
                raise ValueError(f"Country {country} not found in kf model")
            
            self.kfs[country].predict()
            self.kfs[country].update(np.array([prediction]))

            kf_predictions.append(self.kfs[country].x[0])

        return np.array(kf_predictions)
    
    def accurate_predict_update(self, y, countries, noise_var = None):
        """
        Predict the next value and update the Kalman filters with a different noise variance, then revert to the original noise variance.

        y: np.array of shape (n_samples, ) containing the predictions
        countries: np.array of shape (n_samples, ) containing the country of each sample
        override_noise_var: float, the noise variance to use for the update step

        Returns:
        np.array of shape (n_samples, ) containing the predictions
        """
        if noise_var is None:
            noise_var = 1e-7

        if len(countries) != len(np.unique(countries)):
            raise ValueError("There should be one value per country when using accurate_predict_update")

        old_Rs = {}
        for country in countries:
            old_Rs[country] = self.kfs[country].R
            self.kfs[country].R = noise_var

        predictions = self.predict_update(y, countries)

        for country in countries:
            self.kfs[country].R = old_Rs[country]

        return predictions

def apply_kalman_filter(model, preprocessor, use_true_values=False, seed=42, accurate_noise_var=None, accel_var=1e-5):
    """
    Applies a constant acceleration model Kalman filter on the predictions of the model on the high frequency data from the preprocessor.
    Can use true values to correct the Kalman filter state estimate (useful when we have high frequency X and low frequency y).

    model: the neural network model
    preprocessor: the preprocessor object containing the data
    use_true_values: whether to use the true values to correct the Kalman filter estimates
    seed: the seed to use
    accurate_noise_var: the noise variance to use for the accurate data
    accel_var: the acceleration variance to use

    Returns:
    - kf_predictions_melted: the predictions after applying the Kalman filter
    - hf_data_melted: the high frequency data
    - r2: the R^2 of the predictions after applying the Kalman filter
    - smoothness: the smoothness measure of the predictions
    """
    device = get_device(False)
    
    # Based on true GDP data, to be used as low error measurements
    true_data = pd.DataFrame({
        'date': np.concatenate([preprocessor.dates_train, preprocessor.dates_valid], axis=0),
        'country': np.concatenate([preprocessor.country_train, preprocessor.country_valid], axis=0),
        'pred': model(torch.tensor(np.concatenate([preprocessor.X_train, preprocessor.X_valid], axis=0), dtype=torch.float32).to(device)).clone().detach().cpu().numpy().squeeze(),
        'y': np.concatenate([preprocessor.y_train, preprocessor.y_valid], axis=0),
    }).sort_values(by=['date', 'country']).reset_index(drop=True)

    # Remove the duplicates (due to data augmentation)
    true_data = true_data.drop_duplicates(subset=['date', 'country'], keep='first', ignore_index=True)

    # The measurements (high freq predictions) that we want to smooth
    hf_data = pd.DataFrame({
        'date': preprocessor.dates_high_freq,
        'country': preprocessor.country_high_freq,
        'y_pred': model(torch.tensor(preprocessor.x_high_freq, dtype=torch.float32).to(device)).clone().detach().cpu().numpy().squeeze(),
        'Set': 'High Frequency',
    }).sort_values(by=['date', 'country']).reset_index(drop=True)

    # To store the smoothed results
    kf_data = hf_data.copy().assign(y_kf=np.nan)

    # Fit the Kalman filter on the true data (initializes initial state)
    kf = KF()
    kf.fit(true_data['pred'].values, true_data['y'].values, true_data['country'].values, accel_var=accel_var)

    for date in hf_data['date'].unique():
        mask = lambda df: df['date'] == date
        
        true_masked = true_data[mask(true_data)]

        nn_masked = hf_data[mask(hf_data)]
        # Keep only those for which we do not have accurate data
        nn_masked = nn_masked[~nn_masked['country'].isin(true_masked['country'])] if use_true_values else nn_masked

        if use_true_values and true_masked.shape[0] > 0:
            kf_predictions = kf.accurate_predict_update(y=true_masked['y'], countries=true_masked['country'], noise_var=accurate_noise_var)
            kf_data.loc[kf_data['country'].isin(true_masked['country']) & mask(kf_data), 'y_kf'] = kf_predictions
            
        if nn_masked.shape[0] > 0:
            kf_predictions = kf.predict_update(y=nn_masked['y_pred'], countries=nn_masked['country'])
            kf_data.loc[kf_data['country'].isin(nn_masked['country']) & mask(kf_data), 'y_kf'] = kf_predictions

    kf_predictions_melted = kf_data.melt(
        id_vars=["date", "country", "Set"],
        value_vars=["y_kf"], 
        var_name="Type", 
        value_name="Value"
    )

    hf_data_melted = hf_data.melt(
        id_vars=["date", "country", "Set"],
        value_vars=["y_pred"], 
        var_name="Type", 
        value_name="Value"
    )

    # Compute R^2 and smoothness
    r2 = compute_rsquared(true_data['y'].values, true_data.merge(kf_data, on=['date', 'country'], suffixes=('_true', '_kf'))['y_kf'])
    smoothness_loss = measure_smoothness_with_df(kf_data.rename(columns={'y_kf': 'data'}))

    return kf_predictions_melted, hf_data_melted, r2, smoothness_loss