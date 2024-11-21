import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(data, epsilon, number_train, mode=None, all_gdps=None, past_gdp_lags=None):
    """
    Preprocess the data for the prediction model

    Parameters
    ----------
    data : pd.DataFrame
        The data to preprocess
    epsilon : float
        The epsilon value to use to avoid division by zero
    number_train : int
        The number of samples to use for training
    mode : str
        The mode to use for the GDP values, either 'diff' (take the difference) or 'pct' (take the percentage change)
    all_gdps : pd.DataFrame
        The GDP values for all the countries, needed to compute the lagged GDP values
    past_gdp_lags : list
        The list of past GDP lags to include in the data
    """
    if past_gdp_lags:
        if all_gdps is None or any(lag < 1 for lag in past_gdp_lags):
            raise ValueError("You need to provide all the GDP values to include past GDP lags, and the lags should be positive integers")
        else:
            all_gdps['date'] = all_gdps['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

            if mode == 'diff':
                all_gdps['GDP'] = all_gdps.groupby('country')['GDP'].diff()
                all_gdps = all_gdps.dropna()
            elif mode == 'pct':
                all_gdps['GDP'] = all_gdps.groupby('country')['GDP'].pct_change()
                all_gdps = all_gdps.dropna()

    data = data.copy()

    data.rename(columns={'OBS_VALUE': 'GDP'}, inplace=True)
    data.drop(['Reference area'], axis=1, inplace=True)

    data.dropna(inplace=True)

    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data.sort_values('date', inplace=True)
    
    # TODO perform trend removal (on GT and GDP ?)

    if mode == 'diff':
        data['GDP'] = data.groupby('country')['GDP'].diff()
        data = data.dropna()
    elif mode == 'pct':
        data['GDP'] = data.groupby('country')['GDP'].pct_change()
        data = data.dropna()

    if past_gdp_lags:
        for lag in np.sort(past_gdp_lags)[::-1]:
            data[f'GDP_lag_{lag}'] = data.apply(lambda x: _get_lagged_gdp(x['date'], x['country'], all_gdps=all_gdps, lag=lag), axis=1)
        len_before = len(data)
        data.dropna(inplace=True)
        print(f"Dropped {len_before - len(data)} rows because of missing lagged GDP values")

    min_date = data['date'].min()
    data['date'] = (data['date'] - min_date).dt.days

    data_encoded = pd.get_dummies(data, columns=['country'])

    X = data_encoded.drop('GDP', axis=1).reset_index(drop=True)
    y = data_encoded['GDP'].reset_index(drop=True)

    X_train, X_valid  = X.iloc[:number_train], X.iloc[number_train:]
    y_train, y_valid = y.iloc[:number_train], y.iloc[number_train:]

    countries = data['country']
    country_train, country_valid = countries.values[:number_train], countries.values[number_train:],

    return X, y, data['country'], means['GDP'], stds['GDP']
