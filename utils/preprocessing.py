import pandas as pd
import numpy as np
from datetime import datetime

class Preprocessing:
    def __init__(self, data, epsilon, all_GDPs = None, all_GTs = None, mode = None, diff_period = 1, past_GDP_lags = None):
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

    def preprocess_data(self, train_pct):
        """
        Preprocess the data for the prediction model

        Parameters
        ----------
        train_pct : float
            The percentage of data to use for training
        """
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

        data = self.data.copy()
        data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        dates = data['date'].copy()
        
        # TODO perform trend removal (on GT and GDP ?)

        data = _column_to_column_diff(data, 'GDP', 'country', self.mode, self.diff_period, 'date').dropna()

        if self.past_GDP_lags:
            for lag in np.sort(self.past_GDP_lags)[::-1]:
                data[f'GDP_lag_{lag}'] = data.apply(lambda x: _get_lagged_gdp(x['date'], x['country'], all_gdps=all_GDPs, lag=lag), axis=1)
            len_before = len(data)
            data.dropna(inplace=True)
            print(f"Dropped {len_before - len(data)} rows because of missing lagged GDP values")

        self.min_date = data['date'].min()
        data['date'] = (data['date'] - self.min_date).dt.days

        data_encoded = pd.get_dummies(data, columns=['country'], drop_first=True)

        X = data_encoded.drop('GDP', axis=1).reset_index(drop=True)
        y = data_encoded['GDP'].reset_index(drop=True)

        number_train = np.floor(X.shape[0] * train_pct).astype(int)

        self.X_train, self.X_valid  = X.iloc[:number_train], X.iloc[number_train:]
        self.y_train, self.y_valid = y.iloc[:number_train], y.iloc[number_train:]

        countries = data['country']
        self.country_train, self.country_valid = countries.values[:number_train], countries.values[number_train:]
        self.dates_train, self.dates_valid = dates.loc[:number_train], dates.loc[number_train:]


        self.X_means, self.y_mean = self.X_train.mean(), self.y_train.mean()
        self.X_stds, self.y_std = self.X_train.std(), self.y_train.std()

        # replace the mean and std of lagged GDP values by the mean and std of the GDP
        if self.past_GDP_lags:
            self.X_means[self.X_train.columns.str.contains('GDP_lag')] = self.y_mean
            self.X_stds[self.X_train.columns.str.contains('GDP_lag')] = self.y_std

        self.X_train = self._normalize(self.X_train, self.X_means, self.X_stds)
        self.X_valid = self._normalize(self.X_valid, self.X_means,self. X_stds)
        self.y_train = self._normalize(self.y_train, self.y_mean, self.y_std)
        self.y_valid = self._normalize(self.y_valid, self.y_mean, self.y_std)

        print(f"X_train shape : {self.X_train.shape}")
        print(f"X_valid shape : {self.X_valid.shape}")
        print(f"y_train shape : {self.y_train.shape}")
        print(f"y_valid shape : {self.y_valid.shape}")

        return self.X_train, self.y_train, self.X_valid, self.y_valid
    
    def _normalize(self, data, means, stds):
        """
        Normalize the data using the means and stds
        """
        return (data - means) / (stds + self.epsilon)
    
    def get_high_fequency_GT(self):
        """
        Preprocess the Google Trends data for the prediction model
        """
        assert self.X_train is not None, "You need to preprocess the data first"

        data = self.all_GTs.copy()

        # Process the raw GT data
        data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data.sort_values('date', inplace=True)
        data['date'] = (data['date'] - self.min_date).dt.days
        
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

def _column_to_column_diff(data, col_name, grouping_by, mode, diff_period=1, sort_by=None):
    """
    Get the difference or the percentage change of a column

    Parameters
    ----------
    data : pd.DataFrame
        The data to preprocess
    col_name : str
        The name of the column to compute the difference or percentage change
    grouping_by : str
        The column to group by
    mode : str
        The mode to use for the GDP values, either 'diff' (take the difference) or 'pct' (take the percentage change)
    diff_period : int
        The period to use for the difference or percentage change
    sort_by : str
        The column to sort the data by
    """
    if sort_by:
        data.sort_values(sort_by, inplace=True)

    # Get either the difference or the percentage change
    if mode == 'diff':
        data[col_name] = data.groupby(grouping_by)[col_name].diff(periods=diff_period)
    elif mode == 'pct':
        data[col_name] = data.groupby(grouping_by)[col_name].pct_change(periods=diff_period)
    else:
        raise ValueError("The mode should be either 'diff' or 'pct'")
    
    return data
