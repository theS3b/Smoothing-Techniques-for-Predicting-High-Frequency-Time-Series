import pandas as pd

def preprocess_data(data, epsilon):
    data.rename(columns={'OBS_VALUE': 'GDP'}, inplace=True)
    data.drop(['Reference area'], axis=1, inplace=True)
    data.dropna(inplace=True)

    data = pd.get_dummies(data, columns=['country'])

    def date_to_days(date):
        return (date - pd.Timestamp('1970-01-01')).days
    
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].apply(date_to_days)

    means = data.mean()
    stds = data.std()

    data = (data - means) / (stds + epsilon)

    return data.drop('GDP', axis=1), data['GDP'], means['GDP'], stds['GDP']
    