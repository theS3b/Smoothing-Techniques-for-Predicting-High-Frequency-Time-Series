import numpy as np
from filterpy.kalman import KalmanFilter
from models.BaseModel import BaseModel

class KFModel:
    def __init__(self, seed):
        self.seed = seed
        self.nn = BaseModel(seed=seed)

        # KFs for each country
        self.kfs = {}

    def fit(self, X, y, countries):
        self.nn.fit(X, y)

        dt = 1 # Interval between observations
        k = 2e-5 # Tune : increasing make the model closer to the neural network predictions
                 # 0 : flat line
                 # 1e-4 : looks good, maybe not smooth enough
                 # 1e-5 : starts lagging
                 # 1 : follow the neural network predictions

        nn_predictions = self.nn.predict(X)
        nn_var = np.var(nn_predictions - y)

        self.kfs = {}
        for country in np.unique(countries):
            country_data = nn_predictions[countries == country]

            # As initial values, we use the mean of (position, velocity, acceleration) over the first n samples
            n = 5
            initial_gdp = np.mean(country_data[:n])
            initial_velocity = np.mean([country_data[i + 1] - country_data[i] for i in range(n)])
            initial_acceleration = np.mean([country_data[i + 2] - 2 * country_data[i + 1] + country_data[i] for i in range(n)])

            initial_velocity, initial_acceleration = 0, 0

            self.kfs[country] = KalmanFilter(dim_x=3, dim_z=1)
            
            self.kfs[country].x = np.array([initial_gdp, initial_velocity, initial_acceleration])   # Initial state
            self.kfs[country].F = np.array([[1, dt, 0.5 * dt**2],       # position = position + velocity*dt + 0.5*acceleration*dt^2
                                            [0, 1, dt],                 # velocity = velocity + acceleration*dt
                                            [0, 0, 1]])                 # State transition matrix (assuming constant acceleration)
            self.kfs[country].Q = k * np.eye(3)         # Process noise covariance
            self.kfs[country].H = np.array([[1, 0, 0]]) # Measurement matrix, only position is observed
            self.kfs[country].R = nn_var                # Measurement noise covariance, the noise in the neural network predictions
            self.kfs[country].P = np.eye(3)
            # TODO : P and Q (process and measurement noise) should have different values for k 
            # TODO : les KFs sont-ils entrainés séparement

        return self.predict(X, countries)

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