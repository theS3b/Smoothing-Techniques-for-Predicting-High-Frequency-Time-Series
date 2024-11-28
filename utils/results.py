import numpy as np

def compute_rsquared(y_true, y_pred):
    """
    Compute the R-squared value of the predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean = np.mean(y_true)
    ss_total = np.sum((y_true - mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_total