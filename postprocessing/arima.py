from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import pandas as pd

def postprocess_arima(y_pred_train_country, y_pred_valid_country):
    """
    Post-processes the predictions using ARIMA model using sequential integer indexing.

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

    # Calculate residuals
    predictions['residual'] = predictions['y_true'] - predictions['y_pred']

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

        # Extract residuals from the training data
        residuals = train_data['residual'].reset_index(drop=True)

        # Check if we have enough data points to fit ARIMA
        if len(residuals) > 2:
            try:
                model = auto_arima(
                    residuals,
                    start_p=0, max_p=3,
                    start_q=0, max_q=3,
                    start_d=0, max_d=3,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                # Forecast residuals
                n_forecast = len(valid_data)
                forecast_resid = model.predict(n_periods=n_forecast)
                print(f'model params: {model.order}')
                # Adjust predictions
                adjusted_country_data = country_data.copy()
                adjusted_country_data.loc[adjusted_country_data['set'] == 'validation', 'y_pred'] += forecast_resid
            except Exception as e:
                print(f'auto_arima failed for country {country}: {e}')
                adjusted_country_data = country_data.copy()
        else:
            adjusted_country_data = country_data.copy()

        adjusted_predictions.append(adjusted_country_data)

        print(f'Post-processed predictions for {country}')
        print(adjusted_country_data)

    # Combine all adjusted predictions
    adjusted_predictions = pd.concat(adjusted_predictions).sort_index()
    return adjusted_predictions
