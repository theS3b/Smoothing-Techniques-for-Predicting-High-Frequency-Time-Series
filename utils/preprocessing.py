import pandas as pd
from datetime import datetime

def preprocess_data(data, epsilon, take_gdp_diff=True, past_gdp_lag=None):
    """
    Preprocess the data for the prediction model

    Parameters
    ----------
    data : pd.DataFrame
        The data to preprocess
    epsilon : float
        Small number to avoid division by zero
    take_gdp_diff : bool
        Whether to take the difference between two consecutive GDP values
        If true, we drop the first row of the data to avoid NaN values.
    past_gdp_lag : int
        The lag of the GDP values to include in the features.
        If set to None, we don't include any past GDP values.
        Including the value of the GDP at time (now - K) will drop the first K rows of the data.
    """
    data = data.copy()

    data.rename(columns={'OBS_VALUE': 'GDP'}, inplace=True)
    data.drop(['Reference area'], axis=1, inplace=True)

    data.dropna(inplace=True)

    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data['date'] = (data['date'] - data['date'].min()).dt.days

    data.sort_values('date', inplace=True)

    if take_gdp_diff:
        data['GDP'] = data['GDP'].diff()
        data = data[1:]

    if past_gdp_lag:
        data[f'GDP_lag_{past_gdp_lag}'] = data.groupby('country')['GDP'].shift(past_gdp_lag)
        data.dropna(inplace=True)

    # TODO add possibility to include past values of google trends

    data_encoded = pd.get_dummies(data, columns=['country'])

    means = data_encoded.mean()
    stds = data_encoded.std()

    data_encoded = (data_encoded - means) / (stds + epsilon)

    X = data_encoded.drop('GDP', axis=1)
    y = data_encoded['GDP']

    return X, y, data['country'], means['GDP'], stds['GDP']