import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(data, epsilon, train_pct, mode=None, all_gdps=None, past_gdp_lags=None, all_gts=None, past_gt_lags=None):
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
    all_gts : pd.DataFrame
        The Google Trends values for all the countries (not preprocessed), needed to compute the lagged Google Trends values
    past_gt_lags : list
        The list of past Google Trends lags to include in the data
    """
    if past_gdp_lags:
        if all_gdps is None or any(lag < 1 for lag in past_gdp_lags):
            raise ValueError("You need to provide all the GDP values to include past GDP lags, and the lags should be positive integers")
        else:
            all_gdps = all_gdps.copy()

            all_gdps['date'] = all_gdps['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

            if mode == 'diff':
                all_gdps['GDP'] = all_gdps.sort_values('date').groupby('country')['GDP'].diff()
                print("Warning : Dropping the first row of each country because of the diff operation")
            elif mode == 'pct':
                all_gdps['GDP'] = all_gdps.sort_values('date').groupby('country')['GDP'].pct_change()
                print("Warning : Dropping the first row of each country because of the pct operation")

    if past_gt_lags:
        if all_gts is None or any(lag < 1 for lag in past_gt_lags):
            raise ValueError("You need to provide all the Google Trends values to include past Google Trends lags, and the lags should be positive integers")
        else:
            all_gts = all_gts.copy()

            all_gts['date'] = all_gts['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

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
        len_after = len(data)
        if len_before != len_after:
            print(f"Dropped {len_before - len_after} rows because of missing lagged GDP values")

    if past_gt_lags:
        lagged_gts = _get_lagged_gts(all_gts, lags=past_gt_lags)

        data = data.merge(lagged_gts, left_on=["country", "date"], right_on=["country", "date"], how="left")


        len_before = len(data)
        data.dropna(inplace=True)
        len_after = len(data)
        if len_before != len_after:
            print(f"Dropped {len_before - len_after} rows because of missing lagged Google Trends values")

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
    if past_gdp_lags:
        X_means[X_train.columns.str.contains('GDP_lag')] = y_mean
        X_stds[X_train.columns.str.contains('GDP_lag')] = y_std

    # same for lagged GT values
    if past_gt_lags:
        X_means[X_train.columns.str.contains('lag') & ~X_train.columns.str.contains('GDP')] = X_means[X_train.columns.str.contains('average')]
        X_stds[X_train.columns.str.contains('lag') & ~X_train.columns.str.contains('GDP')] = X_stds[X_train.columns.str.contains('average')]

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

def _get_lagged_gts(all_gts, lags):
    """
    Build the lagged Google Trends values
    """
    search_terms = [col for col in all_gts.columns if col.endswith('_average')]

    all_gts[search_terms] = np.log(all_gts[search_terms] + 1)

    for lag in lags:
        diff = (all_gts[search_terms] - all_gts.groupby("country")[search_terms].diff(3 * lag)).add_suffix(f'_lag_{lag}')
        all_gts = pd.concat([all_gts, diff], axis=1)

    all_gts.drop(columns=search_terms, inplace=True)
    all_gts.dropna(inplace=True)

    return all_gts