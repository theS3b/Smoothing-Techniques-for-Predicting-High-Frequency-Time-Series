from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import pandas as pd
import numpy as np

def postprocess_arima_auto(y_pred_train_country, y_pred_valid_country):
    """
    Post-processes the predictions by fitting an ARIMA model on the predictions themselves.

    Args:
        y_pred_train_country (pd.DataFrame): The predictions on the training data.
        y_pred_valid_country (pd.DataFrame): The predictions on the validation data.

    Returns:
        pd.DataFrame: The post-processed predictions.
    """

    # Add a 'set' column to distinguish between training and validation data
    y_pred_train_country = y_pred_train_country.copy()
    y_pred_valid_country = y_pred_valid_country.copy()
    y_pred_train_country['set'] = 'train'
    y_pred_valid_country['set'] = 'validation'

    # Combine predictions
    predictions = pd.concat([y_pred_train_country, y_pred_valid_country])

    adjusted_predictions = []

    # Get the list of unique countries
    countries = predictions['country'].unique()

    for country in countries:
        # Filter data for the current country and sort by 'date' (normalized dates)
        country_data = predictions[predictions['country'] == country].copy()
        country_data.sort_values('date', inplace=True)

        # Split into training and validation sets
        train_data = country_data[country_data['set'] == 'train']
        valid_data = country_data[country_data['set'] == 'validation']

        # Extract predictions from the training data
        train_preds = train_data['y_pred'].reset_index(drop=True)

        # Check if we have enough data points to fit ARIMA
        if len(train_preds) > 2:
            try:
                # Use auto_arima to find the best model
                model = auto_arima(
                    train_preds,
                    start_p=2, max_p=10,
                    start_q=2, max_q=10,
                    max_d=10,
                    d=None,
                    seasonal=False,
                    stationary=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=False,
                    n_jobs=-1,
                    maxiter=100
                )
                # Print best model params
                print(f'Best ARIMA model for {country}: {model.order}')



                # Forecast predictions for the validation period
                n_forecast = len(valid_data)
                forecast_preds = model.predict(n_periods=n_forecast)

                # Adjust predictions
                adjusted_country_data = country_data.copy()

                # Reset index of forecast_preds
                forecast_preds = pd.Series(forecast_preds).reset_index(drop=True)

                # Get validation indices
                validation_indices = adjusted_country_data.loc[adjusted_country_data['set'] == 'validation'].index

                # Ensure lengths match
                if len(forecast_preds) == len(validation_indices):
                    # Replace base model's predictions with ARIMA forecasts
                    adjusted_country_data.loc[validation_indices, 'y_pred'] = forecast_preds.values.astype(np.float32)
                else:
                    print(f'Length of forecasted predictions does not match validation indices for country {country}.')
            except Exception as e:
                print(f'auto_arima failed for country {country}: {e}')
                adjusted_country_data = country_data.copy()
        else:
            adjusted_country_data = country_data.copy()

        adjusted_predictions.append(adjusted_country_data)

    # Combine all adjusted predictions
    adjusted_predictions = pd.concat(adjusted_predictions).sort_index()
    return adjusted_predictions

def postprocess_arima(y_pred_train_country, y_pred_valid_country, p, d, q):
    """
    Post-processes the predictions by fitting an ARIMA model on the predictions themselves.

    Args:
        y_pred_train_country (pd.DataFrame): The predictions on the training data.
        y_pred_valid_country (pd.DataFrame): The predictions on the validation data.
        p (int): The number of lag observations included in the model.
        d (int): The number of times that the raw observations are differenced.
        q (int): The size of the moving average window.

    Returns:
        pd.DataFrame: The post-processed predictions.
    """

    # Add a 'set' column to distinguish between training and validation data
    y_pred_train_country = y_pred_train_country.copy()
    y_pred_valid_country = y_pred_valid_country.copy()
    y_pred_train_country['set'] = 'train'
    y_pred_valid_country['set'] = 'validation'

    # Combine predictions
    predictions = pd.concat([y_pred_train_country, y_pred_valid_country])

    adjusted_predictions = []

    # Get the list of unique countries
    countries = predictions['country'].unique()

    for country in countries:
        # Filter data for the current country and sort by 'date' (normalized dates)
        country_data = predictions[predictions['country'] == country].copy()
        country_data.sort_values('date', inplace=True)

        # Split into training and validation sets
        train_data = country_data[country_data['set'] == 'train']
        valid_data = country_data[country_data['set'] == 'validation']

        # Extract predictions from the training data
        train_preds = train_data['y_pred'].reset_index(drop=True)

        # Check if we have enough data points to fit ARIMA
        if len(train_preds) > 2:
            try:
                # Fit ARIMA model
                model = ARIMA(train_preds, order=(p, d, q))
                model_fit = model.fit()

                # Forecast predictions for the validation period
                n_forecast = len(valid_data)
                forecast_preds = model_fit.forecast(steps=n_forecast)

                # Adjust predictions
                adjusted_country_data = country_data.copy()

                # Reset index of forecast_preds
                forecast_preds = pd.Series(forecast_preds).reset_index(drop=True)

                # Get validation indices
                validation_indices = adjusted_country_data.loc[adjusted_country_data['set'] == 'validation'].index

                # Ensure lengths match
                if len(forecast_preds) == len(validation_indices):
                    # Replace base model's predictions with ARIMA forecasts
                    adjusted_country_data.loc[validation_indices, 'y_pred'] = forecast_preds.values.astype(np.float32)
                else:
                    print(f'Length of forecasted predictions does not match validation indices for country {country}.')
            except Exception as e:
                print(f'ARIMA failed for country {country}: {e}')
                adjusted_country_data = country_data.copy()
        else:
            adjusted_country_data = country_data.copy()

        adjusted_predictions.append(adjusted_country_data)

    # Combine all adjusted predictions
    adjusted_predictions = pd.concat(adjusted_predictions).sort_index()
    return adjusted_predictions


def smooth_nn_predictions_with_arima(pred, p=1, d=1, q=0):
    """
    Smooths predictions from a Neural Network using ARIMA.
    
    Args:
        predictions (np.array or pd.Series): The NN predictions to smooth.
        p (int): AR order (lag order).
        d (int): Difference order.
        q (int): MA order.
        
    Returns:
        np.array: Smoothed predictions.
    """

    predictions = pred.copy()

    # Convert dates
    predictions['date'] = pd.to_datetime(predictions['date'])

    adjusted_predictions = []

    # Get the list of unique countries
    countries = predictions['country'].unique()

    for country in countries:
        # Filter data for the current country and sort by 'date' (normalized dates)
        country_data = predictions[predictions['country'] == country].copy()
        country_data.sort_values('date', inplace=True)

        # Reset index
        country_data.reset_index(drop=True, inplace=True)

        # Check if we have enough data points to fit ARIMA
        if len(country_data) > 2:
            try:
                # Fit ARIMA model
                model = ARIMA(country_data['y_pred_high_freq'], order=(p, d, q))
                model_fit = model.fit()

                # Generate smoothed predictions
                smoothed_predictions = model_fit.fittedvalues

                # Adjust predictions
                adjusted_country_data = country_data.copy()
                adjusted_country_data['y_pred_high_freq'] = smoothed_predictions

            except Exception as e:
                print(f'ARIMA failed for country {country}: {e}')
                adjusted_country_data = country_data.copy()
        else:
            adjusted_country_data = country_data.copy()

        adjusted_predictions.append(adjusted_country_data)

    # Combine all adjusted predictions
    adjusted_predictions = pd.concat(adjusted_predictions).sort_index()
    return adjusted_predictions

def smooth_nn_predictions_with_arima_auto(pred):
    """
    Smooths predictions from a Neural Network using ARIMA.
    
    Args:
        predictions (np.array or pd.Series): The NN predictions to smooth.
        p (int): AR order (lag order).
        d (int): Difference order.
        q (int): MA order.
        
    Returns:
        np.array: Smoothed predictions.
    """

    predictions = pred.copy()

    adjusted_predictions = []

    # Get the list of unique countries
    countries = predictions['country'].unique()

    for country in countries:
        # Filter data for the current country and sort by 'date' (normalized dates)
        country_data = predictions[predictions['country'] == country].copy()
        country_data.sort_values('date', inplace=True)

        # Reset index
        country_data.reset_index(drop=True, inplace=True)

        # Check if we have enough data points to fit ARIMA
        if len(country_data) > 2:
            try:
                # Fit ARIMA model
                model = auto_arima(
                    country_data['y_pred_high_freq'], 
                    seasonal=False,  # Set True if your data has seasonality
                    stepwise=True,   # Stepwise approach to explore parameters
                    suppress_warnings=True,
                    error_action="ignore",  # Ignore errors for convergence issues
                    max_p=16, max_q=16, d=1
                )
                
                # Print best model params
                # print(f'Best ARIMA model for {country}: {model.order}')

                optimal_order = model.order
                arima_model = ARIMA(country_data['y_pred_high_freq'], order=optimal_order)

                # Generate smoothed predictions
                smoothed_predictions = arima_model.fit().fittedvalues

                # Adjust predictions
                adjusted_country_data = country_data.copy()
                adjusted_country_data['y_pred_high_freq'] = smoothed_predictions

            except Exception as e:
                print(f'ARIMA failed for country {country}: {e}')
                adjusted_country_data = country_data.copy()
        else:
            adjusted_country_data = country_data.copy()

        adjusted_predictions.append(adjusted_country_data)

    # Combine all adjusted predictions
    adjusted_predictions = pd.concat(adjusted_predictions).sort_index()
    return adjusted_predictions