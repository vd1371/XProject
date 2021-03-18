#Loading dependencies
import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.ClassificationReport import evaluate_classification

class RandomClassifier(BaseModel):
    
    def __init__(self, name, dl):
        
        super().__init__(name, 'Random', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        self.dl = dl
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        
        self.X, self.Y, _ = dl.load_all()
        
        
    @timeit
    def fit(self):

        value_counts = self.Y.value_counts()
        value_counts = value_counts / value_counts.sum()

        y_pred_train = np.random.choice(value_counts.index, size = len(self.Y_train), p = value_counts.values)
        y_pred_test = np.random.choice(value_counts.index, size = len(self.Y_test), p = value_counts.values)

        evaluate_classification(['OnTrain', self.X_train, self.Y_train, self.dates_train, y_pred_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test, y_pred_test],
                                direc = self.directory,
                                model_name = self.model_name,
                                logger = self.log,
                                slicer = 1)