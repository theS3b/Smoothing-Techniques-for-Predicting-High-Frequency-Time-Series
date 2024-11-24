import numpy as np
from filterpy.kalman import KalmanFilter
from models.MLP import MLP

class KalmanFilterMLP:
    def __init__(self, seed, nb_init_samples=5):
        self.seed = seed
        self.nn = MLP(seed=seed)
        self.nb_init_samples = nb_init_samples

        # KFs for each country
        self.kfs = {}

    def fit(self, X, y, countries):
        self.nn.fit(X, y)

        dt = 1 # Interval between observations
        accel_var = 2e-5 # Acceleration variance, less variance in acceleration means smoother predictions
                # (because it means that the constant accel model is accurate -> more take into account)
                 # 0 : flat line
                 # 1e-4 : looks good, maybe not smooth enough
                 # 1e-5 : starts lagging
                 # 1 : follow the neural network predictions

        nn_predictions = self.nn.predict(X)
        nn_noise_var = np.var(nn_predictions - y)

        self.kfs = {}
        for country in np.unique(countries):
            country_data = nn_predictions[countries == country]

            # As initial values, we use the mean of (position, velocity, acceleration) over the first n samples
            first_n_samples = country_data[:self.nb_init_samples]
            first_n_velocities = [country_data[i + 1] - country_data[i] for i in range(self.nb_init_samples)]
            first_n_accelerations = [country_data[i + 2] - 2 * country_data[i + 1] + country_data[i] for i in range(self.nb_init_samples)]

            initial_gdp, initial_gdp_var = np.mean(first_n_samples), np.var(first_n_samples)
            initial_velocity, initial_velocity_var = np.mean(first_n_velocities), np.var(first_n_velocities)
            initial_acceleration, initial_acceleration_var = np.mean(first_n_accelerations), np.var(first_n_accelerations)

            initial_velocity, initial_acceleration = 0, 0

            self.kfs[country] = KalmanFilter(dim_x=3, dim_z=1)
            
            self.kfs[country].x = np.array([initial_gdp, initial_velocity, initial_acceleration])   # Initial state
            self.kfs[country].F = np.array([[1, dt, 0.5 * dt**2],       # position = position + velocity*dt + 0.5*acceleration*dt^2
                                            [0, 1, dt],                 # velocity = velocity + acceleration*dt
                                            [0, 0, 1]])                 # State transition matrix (assuming constant acceleration)
            self.kfs[country].Q = accel_var * np.eye(3)         # Prediction (constant acceleration model) noise covariance
            self.kfs[country].H = np.array([[1, 0, 0]]) # Measurement matrix, only position is observed
            self.kfs[country].R = nn_noise_var                # Measurement noise covariance, the noise in the neural network predictions
            self.kfs[country].P = np.diag([             # State covariance
                initial_gdp_var,
                initial_velocity_var,
                initial_acceleration_var
            ])

    def predict(self, X, countries):
        model_predictions = self.nn.predict(X)

        kf_predictions = []
        for prediction, country in zip(model_predictions, countries):
            if country not in self.kfs:
                raise ValueError(f"Country {country} not found in kf model")
            
            self.kfs[country].predict()
            self.kfs[country].update(np.array([prediction]))

            kf_predictions.append(self.kfs[country].x[0])

        return np.array(kf_predictions)