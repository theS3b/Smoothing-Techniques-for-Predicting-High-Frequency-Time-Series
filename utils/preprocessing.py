import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(data, epsilon, train_pct, mode=None, diff_period=1, all_GDPs=None, past_GDP_lags=None):
    """
    Preprocess the data for the prediction model

    Parameters
    ----------
    data : pd.DataFrame
        The data to preprocess
    epsilon : float
        The epsilon value to use to avoid division by zero
    train_pct : float
        The percentage of data to use for training
    mode : str
        The mode to use for the GDP values, either 'diff' (take the difference) or 'pct' (take the percentage change)
    diff_period : int
        The period to use for the difference or percentage change
    all_GDPs : pd.DataFrame
        The GDP values for all the countries, needed to compute the lagged GDP values
    past_GDP_lags : list
        The list of past GDP lags to include in the data
    """

    # Check if we need the to include past GDP values
    if past_GDP_lags:
        if all_GDPs is None or any(lag < 1 for lag in past_GDP_lags):
            raise ValueError("You need to provide all the GDP values to include past GDP lags, and the lags should be positive integers")
        
        # Copy for good practice
        all_GDPs = all_GDPs.copy()

        # Convert the date to datetime
        all_GDPs['date'] = all_GDPs['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

        # Get either the difference or the percentage change
        all_GDPs = _column_to_column_diff(all_GDPs, 'GDP', 'country', mode, diff_period, 'date')

    data = data.copy()
    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    
    # TODO perform trend removal (on GT and GDP ?)

    data = _column_to_column_diff(data, 'GDP', 'country', mode, diff_period, 'date').dropna()

    if past_GDP_lags:
        for lag in np.sort(past_GDP_lags)[::-1]:
            data[f'GDP_lag_{lag}'] = data.apply(lambda x: _get_lagged_gdp(x['date'], x['country'], all_gdps=all_GDPs, lag=lag), axis=1)
        len_before = len(data)
        data.dropna(inplace=True)
        print(f"Dropped {len_before - len(data)} rows because of missing lagged GDP values")

    min_date = data['date'].min()
    data['date'] = (data['date'] - min_date).dt.days

    data_encoded = pd.get_dummies(data, columns=['country'])

    X = data_encoded.drop('GDP', axis=1).reset_index(drop=True)
    y = data_encoded['GDP'].reset_index(drop=True)

    number_train = np.floor(X.shape[0] * train_pct).astype(int)

    X_train, X_valid  = X.iloc[:number_train], X.iloc[number_train:]
    y_train, y_valid = y.iloc[:number_train], y.iloc[number_train:]

    countries = data['country']
    country_train, country_valid = countries.values[:number_train], countries.values[number_train:],

    X_means, y_mean = X_train.mean(), y_train.mean()
    X_stds, y_std = X_train.std(), y_train.std()

    # replace the mean and std of lagged GDP values by the mean and std of the GDP
    if past_GDP_lags:
        X_means[X_train.columns.str.contains('GDP_lag')] = y_mean
        X_stds[X_train.columns.str.contains('GDP_lag')] = y_std

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


    X_valid = _normalize(X, X_means, X_stds, epsilon)

    print(f"New X_valid shape : {X_valid.shape}")

    return X_valid, countries

def _normalize(data, means, stds, epsilon):
    """
    Normalize the data using the means and stds
    """
    return (data - means) / (stds + epsilon)

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
