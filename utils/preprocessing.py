import pandas as pd
from datetime import datetime

def preprocess_data(data, epsilon):
    data.rename(columns={'OBS_VALUE': 'GDP'}, inplace=True)
    data.drop(['Reference area'], axis=1, inplace=True)
    data.dropna(inplace=True)

    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data['date'] = (data['date'] - data['date'].min()).dt.days
    data['date'] = data['date'] / data['date'].max()

    data.sort_values('date', inplace=True)

    data_encoded = pd.get_dummies(data, columns=['country'])
    
    means = data_encoded.mean()
    stds = data_encoded.std()

    data_encoded = (data_encoded - means) / (stds + epsilon)

    return data_encoded.drop('GDP', axis=1), data_encoded['GDP'], means['GDP'], stds['GDP']
    