import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
import utils.downward_trend_removal as dtr

class Preprocessing:
    PLOT_GT_REMOVAL = 'plot_gt_removal'

    def __init__(self, epsilon, all_GDPs, all_GTs, gdp_diff_period = 1, seed = 42):
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
        self.epsilon = epsilon
        self.gdp_diff_period = gdp_diff_period
        self.all_GDPs = all_GDPs
        self.all_GTs = all_GTs
        self.seed = seed

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

    def preprocess_data(self, train_pct, gt_trend_removal, mode, 
                        gt_data_transformations=[],
                        noisy_data_stds=[], 
                        keep_pca_components = -1, 
                        add_encoded_month=True, 
                        other_params={}):
        """
        Preprocess the data for the prediction model

        Parameters
        ----------
        train_pct : float
            The percentage of data to use for training
        """
        np.random.seed(self.seed)
        
        ### ALL GDPs
        # Copy for good practice
        all_GDPs = self.all_GDPs.copy()

        # Get either the difference or the percentage change
        all_GDPs = _column_to_column_diff(all_GDPs, 'GDP', 'country', mode, self.gdp_diff_period, 'date')

        ### All GTs
        if gt_trend_removal:
            all_GTs = dtr.detrend_gts(all_GTs, plot=(Preprocessing.PLOT_GT_REMOVAL in other_params))
        else:
            all_GTs = self.all_GTs.copy()

        # Do custom transformations of the GT data
        for gt_data_transformation in gt_data_transformations:
            all_GTs = gt_data_transformation(all_GTs)

        # Join the GTs and GDPs
        data = pd.merge(
            left=all_GTs, 
            right=all_GDPs,
            how='inner',
            left_on=['country', 'date'],
            right_on=['country', 'date'],
        )

        data.dropna(inplace=True)
        data.sort_values(['country', 'date'], inplace=True)  # Sort the data for better clarity

        # Encode dummy variables
        data_encoded = pd.get_dummies(data, columns=['country'], drop_first=True)

        # Separate X and y
        X = data_encoded.drop('GDP', axis=1).reset_index(drop=True)
        y = data_encoded['GDP'].reset_index(drop=True)

        X['date'] = pd.to_datetime(X['date'])

        # Determine the splitting date
        unique_dates = sorted(X['date'].unique())  # Very important to sort the dates here
        splitting_date_calc = unique_dates[int(train_pct * len(unique_dates))]

        train_elems = X['date'] < splitting_date_calc
        valid_elems = X['date'] >= splitting_date_calc

        # Store dates and countries 
        print(X.shape, data.shape)

        self.dates_train, self.dates_valid = X[train_elems]['date'], X[valid_elems]['date']
        self.country_train, self.country_valid = data.mask(train_elems)['country'], data.mask(valid_elems)['country']

        # We don't want the date in the training set
        X.drop('date', axis=1, inplace=True)

        # Split the data
        X_train, X_valid  = X[train_elems], X[valid_elems]
        y_train, y_valid = y[train_elems], y[valid_elems]

        # Prepare the normalization (note that we use only the training data for the mean and std)
        self.X_means, self.y_mean = X_train.mean(), y_train.mean()
        self.X_stds, self.y_std = X_train.std(), y_train.std()

        # Normalize the data, note that we use the mean and std of the training data for normalization
        X_train = self._normalize(X_train, self.X_means, self.X_stds)
        X_valid = self._normalize(X_valid, self.X_means, self.X_stds)
        y_train = self._normalize(y_train, self.y_mean, self.y_std)
        y_valid = self._normalize(y_valid, self.y_mean, self.y_std)

        # Add the month of the date as a feature (without normalizing it, so that it can be played with (e.g. maybe a use it for interpolation))
        if add_encoded_month:
            X_train["month"] = self.dates_train.apply(lambda x: x.month)
            X_valid["month"] = self.dates_valid.apply(lambda x: x.month)
            X_train = pd.get_dummies(X_train, columns=["month"], dtype=float)
            X_valid = pd.get_dummies(X_valid, columns=["month"], dtype=float)

        # Do PCA if requested
        if keep_pca_components > 0:
            pca_model = PCA(keep_pca_components)
            X_train_np = pca_model.fit_transform(X_train)
            X_valid_np = pca_model.transform(X_valid)
        else:
            X_train_np = X_train.values
            X_valid_np = X_valid.values

        y_train_np = y_train.values
        y_valid_np = y_valid.values

        # Add noisy points if requested
        if noisy_data_stds:
            noisy_data = []
            for std in noisy_data_stds:
                noisy_data.append(X_train_np + np.random.normal(0, std, X_train_np.shape))

            X_train_np = np.concatenate([X_train_np] + noisy_data, axis=0)
            y_train_np = np.concatenate([y_train_np] * (len(noisy_data) + 1), axis=0)

        # Shuffle the training data
        shuffle_train = np.random.permutation(X_train_np.shape[0])

        # Store the data
        self.X_train = X_train_np[shuffle_train]
        self.y_train = y_train_np[shuffle_train]
        self.dates_train = self.dates_train.iloc[shuffle_train]
        self.country_train = self.country_train.iloc[shuffle_train]
        self.X_valid = X_valid_np
        self.y_valid = y_valid_np

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
        pass
    
    return data
