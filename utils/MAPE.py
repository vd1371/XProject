import numpy as np

def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if np.all(y_true):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        return np.nan