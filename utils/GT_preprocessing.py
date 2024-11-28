from load_data import load_data
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt

# Load data
df, all_gdps = load_data()

# Convert 'date' to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Identify search term columns
search_terms = [col for col in df.columns if col.endswith('_average')]

# Drop rows with missing values
df.dropna(inplace=True)

# Create a time index
df['time_index'] = np.arange(len(df))

# Initialize dictionaries to store new columns
log_columns = {}
detrended_columns = {}

# Replace zeros and take log transformation
for term in search_terms:
    df[term].replace(0, np.nan, inplace=True)
    df[term].fillna(1, inplace=True)  # Replace NaNs with 1 to avoid log(0)
    log_columns[f'log_{term}'] = np.log(df[term])

# Convert log_columns dictionary to DataFrame and concatenate
df_log = pd.DataFrame(log_columns, index=df.index)
df = pd.concat([df, df_log], axis=1)

# Detrend the log-transformed search terms
for term in search_terms:
    X = df['time_index'].values.reshape(-1, 1)
    y = df[f'log_{term}'].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    detrended_columns[f'detrended_{term}'] = df[f'log_{term}'] - trend.flatten()

# Convert detrended_columns dictionary to DataFrame and concatenate
df_detrended = pd.DataFrame(detrended_columns, index=df.index)
df = pd.concat([df, df_detrended], axis=1)

# Create week-of-year feature
df['week_of_year'] = df.index.isocalendar().week.astype(int)

# Create week dummies
week_dummies = pd.get_dummies(df['week_of_year'], prefix='week', drop_first=True)

# Regularization parameter (alpha)
alpha_value = 1.0  # Adjust based on cross-validation or domain knowledge

# Initialize dictionaries to store seasonal effects and adjusted terms
seasonal_effects = {}
seasonally_adjusted = {}

# Perform seasonal adjustment using Ridge regression
for term in search_terms:
    y = df[f'detrended_{term}'].values
    X = week_dummies.values

    # Fit Ridge Regression
    ridge = Ridge(alpha=alpha_value, fit_intercept=False)
    ridge.fit(X, y)
    seasonal_effect = ridge.predict(X)
    
    # Store seasonal effect and seasonally adjusted term
    seasonal_effects[f'seasonal_{term}'] = seasonal_effect
    seasonally_adjusted[f'seasonally_adjusted_{term}'] = y - seasonal_effect

# Convert dictionaries to DataFrames and concatenate
df_seasonal_effects = pd.DataFrame(seasonal_effects, index=df.index)
df_seasonally_adjusted = pd.DataFrame(seasonally_adjusted, index=df.index)
df = pd.concat([df, df_seasonal_effects, df_seasonally_adjusted], axis=1)

# Plot seasonally adjusted series
for term in search_terms:
    df[f'seasonally_adjusted_{term}'].plot(label=term)
    plt.title(f'Seasonally Adjusted {term}')
    plt.legend()
    plt.show()
