import numpy as np

def R2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]**2