import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9
from .AwesomeLogger import Logger

class BaseModel(object):
    def __init__(self,  model_name = 'Unknown',
                        data_loader = None):
        super().__init__()

        self.project_name = data_loader.project_name
        self.file_name = data_loader.file_name
        self.model_name = model_name
        self.directory = os.path.join("./XReport", self.project_name, self.file_name, model_name)
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        logging_address = os.path.join(self.directory, 'Report.log')
        self.log = Logger(logger_name = f'{self.project_name}-{self.file_name}-{model_name}-Logger',
                            address = logging_address,
                            mode='a')

        self.dl = data_loader
        self.dl.log_description(self.log)
