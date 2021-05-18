import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9
from utils.AwesomeLogger import Logger

class BaseModel(object):
    def __init__(self, name = None, model_name = 'Unknown', data_loader = None, direc = None):
        super().__init__()

        self.name = name
        self.model_name = model_name
        self.directory = os.path.join("./XReport", self.name, model_name)
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        logging_address = os.path.join(self.directory, 'Report.log')
        self.log = Logger(logger_name = f'{self.name}-{model_name}-Logger', address = logging_address , mode='a')

        self.dl = data_loader
        self.dl.log_description(self.log)


def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if np.all(y_true):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        return np.nan

def R2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]**2

def CorCoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]