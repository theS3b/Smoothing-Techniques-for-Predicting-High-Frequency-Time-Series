import pandas as pd
import os

DATA_PATH = 'data'
GDP_PATH = os.path.join(DATA_PATH, 'gdp/GDP Data.csv')
GOOGLE_TRENDS_BASE_PATH = os.path.join(DATA_PATH, 'google_trends/trends_data_by_topic_')

"""
The model is trained on all the countries in the dataset, using dummy variables :

"This  paper  uses  a  neural  panel  model,  which  exploits  a  large  sample  of  observations  from  
46 countries while capturing cross-country heterogeneity. Neural networks are able to handle 
heterogeneity in the data as long as country dummies are included. A neural network whose architecture 
incudes  an intermediate  layer  with  enough neurons  (in our  case,  100) can  flexibly  model  each  possible  
interaction between Google Trends variables and country dummies. Each neuron takes as input signals 
from Google Trends variables and country dummies, and returns a non-linear function of the weighted sum 
of these inputs. As a result, the model can capture country-specific elasticities."
(Tracking activity in real time with Google Trends, Nicolas Woloszko)
"""

id_to_name = {
    'CH' : 'Switzerland',
    'DE' : 'Germany',
    'GB' : 'United Kingdom',
    'JP' : 'Japan',
    'CA' : 'Canada',
    'KR' : 'Korea',
    'US' : 'United States',
}

Q_lookup = {
    'Q1' : 3,
    'Q2' : 6,
    'Q3' : 9,
    'Q4' : 12,
}

def load_data():
    # Load GDP data
    gdp_data = pd.read_csv(GDP_PATH)
    gdp_data = gdp_data[gdp_data['Price base']=='Current prices']
    y = gdp_data[['Reference area','TIME_PERIOD','OBS_VALUE']]
    del gdp_data

    # Load GT data
    all_GT_data = load_gt_data()
    
    # Match the date format
    y['date'] = y['TIME_PERIOD'].apply(lambda x: f"{x.split('-')[0]}-{Q_lookup[x.split('-')[1]]:02d}-01")
    y.drop('TIME_PERIOD', inplace=True, axis=1)

    # Rename columns
    y.rename(columns={'Reference area':'country', 'OBS_VALUE':'GDP'}, inplace=True)

    # Inner join on GDP and GT data (on country and date)
    data = pd.merge(
        left=all_GT_data, 
        right=y,
        how='inner',
        left_on=['country', 'date'],
        right_on=['country', 'date'],
    )

    # All data, with dropna for safety
    dataset = data.dropna()

    # Only the countries present in the GT data (with have lots countries for the GDP data but few for the GT data)
    # This also contains past GDP values (before the GT data starts)
    all_GDPs = y[y['country'].isin(all_GT_data['country'].unique())].dropna()

    return dataset, all_GDPs, all_GT_data

def load_gt_data():
    X = []
    for id_ in id_to_name.keys():
        f = pd.read_csv(f'{GOOGLE_TRENDS_BASE_PATH}{id_}.csv')
        cols_to_keep = [c for c in f.columns if c[-8:]=='_average' or c=='date']
        f = f[cols_to_keep]
        f['country'] = id_to_name[id_]
        f.columns = [c.replace(f'{id_}_', '') for c in f.columns]
        X.append(f)
    X = pd.concat(X).drop(['Trademark_attorney_average', 
                                        'Grants-in-Aid_for_Scientific_Research_average',
                                        'Research_&_Experimentation_Tax_Credit_average'],axis=1).reset_index(drop=True)

    return X.dropna()