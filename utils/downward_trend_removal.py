import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.decomposition import PCA

def detrend_gts(data, plot=False):
    data = data.copy()
    
    gt_cols = [col for col in data.columns if col.endswith('_average')]

    if plot:
        plt.figure(figsize=(15, 4))
    
    # Compute the log of the mean SVI values across countries
    means_over_countries = data.drop('country', axis=1).groupby('date').mean()
    log_svi = means_over_countries.apply(np.log1p)

    mean_over_time = log_svi.mean(axis=1)
    log_svi_average_mean, log_svi_average_std = mean_over_time.mean(), mean_over_time.std()

    if plot:
        plt.subplot(131)
        plt.plot(mean_over_time)
        plt.title('Mean of the mean log-SVI values\nacross countries and categories')

    # Apply HP filtering to extract long-term trend
    hp_output = log_svi.apply(lambda x: hpfilter(x, lamb=1600)[1])  # Long-term trend
    if plot:
        plt.subplot(132)
        plt.plot(hp_output)
        plt.title('High-pass filter output')

    # Normalize the long-term trends for PCA
    mean, std = hp_output.mean(axis=1).values, hp_output.std(axis=1).values
    hp_output = (hp_output - mean[:, None]) / std[:, None]

    # Apply PCA to extract the common component
    pca = PCA(n_components=1)
    common_component = pca.fit_transform(hp_output)

    # Rescale PCA component to the same scale as log-SVI
    common_component = (common_component - np.mean(common_component)) / np.std(common_component)
    common_component = common_component * log_svi_average_std + log_svi_average_mean  # Rescaled to match log-SVI

    if plot:
        plt.subplot(133)
        plt.plot(common_component)
        plt.title('Common component extracted by PCA,\nrescaled to match log-SVI')
        plt.show()

    # Subtract the common component from each series
    zero_mask = data[gt_cols] != 0
    data[gt_cols] = np.log1p(data[gt_cols])
    unique_dates = data['date'].unique()
    common_component_map = pd.Series(common_component.ravel(), index=unique_dates)
    
    data['common_component'] = data['date'].map(common_component_map)
    data[gt_cols] = data[gt_cols].sub(data['common_component'], axis=0)
    data.drop(columns=['common_component'], inplace=True)

    # Return to original SVI scale
    data[gt_cols] = np.expm1(data[gt_cols])
    gt_min, gt_max = np.min(data[gt_cols].values), np.max(data[gt_cols].values)
    data[gt_cols] = (data[gt_cols]) / (gt_max - gt_min) * 100
    data[gt_cols] = data[gt_cols].where(zero_mask, 0) # Keep zeros (no data)

    return data