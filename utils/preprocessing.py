import pandas as pd
import numpy as np
from datetime import datetime

class Preprocessing:
    def __init__(self, data, epsilon, all_GDPs = None, all_GTs = None, mode = None, diff_period = 1, past_GDP_lags = None, seed = 42):
        """
        Parameters
        ----------
        epsilon : float
            The epsilon value to use to avoid division by zero
        mode : str
            The mode to use for the GDP values, either 'diff' (take the difference) or 'pct' (take the percentage change)
        diff_period : int
            The period to use for the difference or percentage change
        all_GDPs : pd.DataFrame
            The GDP values for all the countries, needed to compute the lagged GDP values
        past_GDP_lags : list
            The list of past GDP lags to include in the data
        """
        self.data = data
        self.epsilon = epsilon
        self.mode = mode
        self.diff_period = diff_period
        self.all_GDPs = all_GDPs
        self.past_GDP_lags = past_GDP_lags
        self.all_GTs = all_GTs
        self.seed = seed

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

        self.country_train = None
        self.country_valid = None
        self.dates_train = None
        self.dates_valid = None

        self.X_means = None
        self.X_stds = None
        self.y_mean = None
        self.y_std = None

        self.min_date = None

        self.preprocessed_high_freq_gt = None

    def preprocess_data(self, train_pct, shuffle=False, splitting_date=None):
        """
        Preprocess the data for the prediction model

        Parameters
        ----------
        train_pct : float
            The percentage of data to use for training
        """
        np.random.seed(self.seed)

        # Check if we need the to include past GDP values
        if self.past_GDP_lags:
            if all_GDPs is None or any(lag < 1 for lag in self.past_GDP_lags):
                raise ValueError("You need to provide all the GDP values to include past GDP lags, and the lags should be positive integers")
            
            # Copy for good practice
            all_GDPs = all_GDPs.copy()

            # Convert the date to datetime
            all_GDPs['date'] = all_GDPs['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

            # Get either the difference or the percentage change
            all_GDPs = _column_to_column_diff(all_GDPs, 'GDP', 'country', self.mode, self.diff_period, 'date')

        # Copy for good practice
        data = self.data.copy()

        # Convert the date to datetime
        data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        
        # TODO perform trend removal (on GT and GDP ?)

        data = _column_to_column_diff(data, 'GDP', 'country', self.mode, self.diff_period, 'date').dropna()

        if self.past_GDP_lags:
            for lag in np.sort(self.past_GDP_lags)[::-1]:
                data[f'GDP_lag_{lag}'] = data.apply(lambda x: _get_lagged_gdp(x['date'], x['country'], all_gdps=all_GDPs, lag=lag), axis=1)
            len_before = len(data)
            data.dropna(inplace=True)
            print(f"Dropped {len_before - len(data)} rows because of missing lagged GDP values")

        data_encoded = pd.get_dummies(data, columns=['country'], drop_first=True)

        if shuffle:
            shuffle_idx = np.random.permutation(data_encoded.shape[0])
        else:
            shuffle_idx = np.arange(data_encoded.shape[0])

        # Separate X and y
        X = data_encoded.drop('GDP', axis=1).reset_index(drop=True).iloc[shuffle_idx]
        y = data_encoded['GDP'].reset_index(drop=True).iloc[shuffle_idx]

        countries = data['country']

        if shuffle:
            number_train = np.floor(X.shape[0] * train_pct).astype(int)

            # Store dates and remove them from the data (using iloc for positional slicing)
            self.dates_train, self.dates_valid = X.iloc[:number_train]['date'], X.iloc[number_train:]['date']
            X.drop('date', axis=1, inplace=True)

            # Split the data
            self.X_train, self.X_valid  = X.iloc[:number_train], X.iloc[number_train:]
            self.y_train, self.y_valid = y.iloc[:number_train], y.iloc[number_train:]

            # Store the countries separately (otherwise we would have to work with the one-hot encoded countries)
            self.country_train, self.country_valid = countries.iloc[shuffle_idx].values[:number_train], countries.iloc[shuffle_idx].values[number_train:]

        else:
            unique_dates = X['date'].unique()
            splitting_date_calc = splitting_date if splitting_date is not None else unique_dates[int(train_pct * len(unique_dates))]

            train_elems = X['date'] < splitting_date_calc
            valid_elems = X['date'] >= splitting_date_calc

            # Store dates and remove them from the data (using iloc for positional slicing)
            self.dates_train, self.dates_valid = X[train_elems]['date'], X[valid_elems]['date']
            X.drop('date', axis=1, inplace=True)

            # Split the data
            self.X_train, self.X_valid  = X[train_elems], X[valid_elems]
            self.y_train, self.y_valid = y[train_elems], y[valid_elems]

            number_train = len(self.X_train)

            # Store the countries separately (otherwise we would have to work with the one-hot encoded countries)
            self.country_train, self.country_valid = countries.iloc[shuffle_idx].values[:number_train], countries.iloc[shuffle_idx].values[number_train:]

        # Prepare the normalization
        self.X_means, self.y_mean = self.X_train.mean(), self.y_train.mean()
        self.X_stds, self.y_std = self.X_train.std(), self.y_train.std()

        # replace the mean and std of lagged GDP values by the mean and std of the GDP
        if self.past_GDP_lags:
            self.X_means[self.X_train.columns.str.contains('GDP_lag')] = self.y_mean
            self.X_stds[self.X_train.columns.str.contains('GDP_lag')] = self.y_std

    X_train = _normalize(X_train, X_means, X_stds, epsilon)
    X_valid = _normalize(X_valid, X_means, X_stds, epsilon)
    y_train = _normalize(y_train, y_mean, y_std, epsilon)
    y_valid = _normalize(y_valid, y_mean, y_std, epsilon)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_valid shape : {X_valid.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_valid shape : {y_valid.shape}")

    return X_train, y_train, X_valid, y_valid, country_train, country_valid, X_means, X_stds, y_mean, y_std, min_date

def preprocess_gt_data(data, epsilon, X_means, X_stds, min_date):
    """
    Preprocess the Google Trends data for the prediction model

    Parameters
    ----------
    data : pd.DataFrame
        The data to preprocess
    epsilon : float
        The epsilon value to use to avoid division by zero
    X_means : pd.Series
        The means of the training data
    X_stds : pd.Series
        The standard deviations of the training data
    """
    data = data.copy()

    data.dropna(inplace=True)

    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data.sort_values('date', inplace=True)
    
    data['date'] = (data['date'] - min_date).dt.days
    
    countries = data['country'].reset_index(drop=True)

    data_encoded = pd.get_dummies(data, columns=['country'])

        X = data_encoded.reset_index(drop=True)
        X_normed = self._normalize(X, self.X_means, self.X_stds)

        print(f"High Frequency GT shape : {X_normed.shape}")
        return X_normed, countries


def _get_lagged_gdp(date, country, all_gdps, lag):
    """
    Build the lagged GDP values
    """
    all_dates = all_gdps[all_gdps['country'] == country]['date'].sort_values().values

    curr_date_index = np.where(all_dates == date)

    if not curr_date_index or curr_date_index[0][0] < lag:
        print(f"Warning : {country} has not enough data to compute the lagged GDP at date {date} with lag {lag}, removing the row.")
        return np.nan

    date_lag = all_dates[curr_date_index[0][0] - lag]

    return all_gdps[(all_gdps['date'] == date_lag) & (all_gdps['country'] == country)]['GDP'].values[0]