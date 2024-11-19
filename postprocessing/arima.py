from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def postprocess_arima(y_pred_train_country, y_pred_valid_country, p, d, q):
    """
    Post-processes the predictions using ARIMA model using sequential integer indexing.

    Args:
        y_pred_train_country (pd.DataFrame): The predictions on the training data.
        y_pred_valid_country (pd.DataFrame): The predictions on the validation data.
        p (int): The AR order.
        d (int): The differencing order.
        q (int): The MA order.
    
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

        # Split into training and validation sets
        train_data = country_data[country_data['set'] == 'train']
        valid_data = country_data[country_data['set'] == 'validation']

        # Extract residuals from the training data
        residuals = train_data['residual'].reset_index(drop=True)

        # Check if we have enough data points to fit ARIMA
        if len(residuals) > max(p, d, q):
            # Fit ARIMA model using integer index as time index
            model = ARIMA(residuals, order=(p, d, q))
            model_fit = model.fit()

            # Forecast residuals for the validation period
            n_forecast = len(valid_data)
            print(f"Forecasting {n_forecast} steps for {country}")
            forecast_resid = model_fit.forecast(steps=n_forecast)

            # Adjust the predictions in the validation set
            adjusted_country_data = country_data.copy()
            adjusted_country_data.loc[adjusted_country_data['set'] == 'validation', 'y_pred'] += forecast_resid.values

        else:
            # If not enough data, keep the original predictions
            adjusted_country_data = country_data.copy()

        adjusted_predictions.append(adjusted_country_data)

    # Combine all adjusted predictions
    adjusted_predictions = pd.concat(adjusted_predictions).sort_index()
    return adjusted_predictions
