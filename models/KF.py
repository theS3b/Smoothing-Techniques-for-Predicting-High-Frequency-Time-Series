import numpy as np
from filterpy.kalman import KalmanFilter
from models.MLP import MLP

class KF:
    def __init__(self, nb_init_samples=5):
        self.nb_init_samples = nb_init_samples

        # KFs for each country
        self.kfs = {}

    def fit(self, y_pred, y_true, countries, accel_var=1e-4):
        """
        Fit the Kalman filters to the data
        
        y_pred: np.array of shape (n_samples, ) containing the predictions
        y_true: np.array of shape (n_samples, ) containing the true values
        countries: np.array of shape (n_samples, ) containing the country of each sample
        theoretical_noise_var: float, the theoretical model noise variance, less variance in means smoother predictions for a constant acceleration model.
                Default to 1e-4.
                    1 : follow the neural network predictions
                    1e-4 : smoother but close to the neural network predictions
                    1e-5 : starts lagging
                    0 : flat line
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

    def fit_predict(self, y_pred, y_true, countries):
        self.fit(y_pred, y_true, countries)

        return self.predict_update(y_pred, countries)

    def predict_update(self, y, countries):
        kf_predictions = []
        for prediction, country in zip(y, countries):
            if country not in self.kfs:
                raise ValueError(f"Country {country} not found in kf model")
            
            self.kfs[country].predict()
            self.kfs[country].update(np.array([prediction]))

            kf_predictions.append(self.kfs[country].x[0])

        return np.array(kf_predictions)
    
    def accurate_predict_update(self, y, countries, override_noise_var = None):
        if override_noise_var is None:
            override_noise_var = 1e-7

        if len(countries) != len(np.unique(countries)):
            raise ValueError("There should be one value per country when using accurate_predict_update")

        old_Rs = {}
        for country in countries:
            old_Rs[country] = self.kfs[country].R
            self.kfs[country].R = override_noise_var

        predictions = self.predict_update(y, countries)

        for country in countries:
            self.kfs[country].R = old_Rs[country]

        return predictions
