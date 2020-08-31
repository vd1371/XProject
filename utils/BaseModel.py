import os
import numpy as np
import matplotlib.pyplot as plt

from utils.AwesomeLogger import Logger


class BaseModel(object):
    def __init__(self, name = None, model_name = 'Unknown'):
        super().__init__()

        self.name = name        
        self.directory = os.path.join("./XReport", self.name, model_name)
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        logging_address = os.path.join(self.directory, 'Report.log')
        self.log = Logger(logger_name = self.name + '-Logger', address = logging_address , mode='a')


def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except ZeroDivisionError:
        return 0

def R2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]**2

def CorCoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]