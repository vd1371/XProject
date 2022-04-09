import numpy as np

def CorCoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]