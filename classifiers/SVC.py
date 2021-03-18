#Loading dependencies
import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.ClassificationReport import evaluate_classification
from utils.FeatureImportanceReport import report_feature_importance
from utils.SpecialPlotters import train_cv_analyzer_plotter

from sklearn.svm import SVC
 
class SVMC(BaseModel):
        
    def __init__(self, name, dl):
        
        super().__init__(name, 'SVC', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        self.dl = dl
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        
        self.X, self.Y, _ = dl.load_all()

    def set_params(self, **params):
        
        self.kernel = params.pop('kernel', 'rbf') 
        self.gamma = params.pop('gamma', 'auto')
        self.C = params.pop('C', 1)

    def log_params(self):

        self.log.info(pprint.pformat({
            "Model_type": 'SVC',
            'kernel': self.kernel,
            'gamma': self.gamma,
            'C': self.C,
            'random_state': self.dl.random_state
            }))
    
    @timeit
    def fit(self):

        self.set_params()
        self.log_params()
        
        self.model = SVC(C = self.C,
                        kernel= self.kernel,
                        gamma = self.gamma)
        self.model.fit(self.X_train, self.Y_train)

        evaluate_classification(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = self.model,
                                model_name = self.model_name,
                                logger = self.log,
                                slicer = 1)
        
        joblib.dump(self.model, self.directory + f"/{self.model_name}.pkl")
        
        # Plotting the Importances
        if self.kernel == 'linear':
            report_feature_importance(self.directory,
                                    self.model.coef_[0],
                                    self.X_train.columns,
                                    self.n_top_features,
                                    self.model_name,
                                    self.log)
    
    @timeit
    def regularization_analysis(self, start = 1, end = 100000, step = 2):

        train_error, cv_error = [], []
        xticks = []
        
        i = start
        while i < end:
            
            xticks.append(i)
            model = SVR(C=C, kernel= "rbf", epsilon = i, gamma = 'auto')
            model.fit(self.X_train, self.Y_train)
            
            train_error.append(mean_squared_error(self.Y_train, model.predict(self.X_train)))
            cv_error.append(mean_squared_error(self.Y_test, model.predict(self.X_test)))
            
            print (f"Step {i} of SVM epsilon_analysis is done")
        
            i = i *step

        train_cv_analyzer_plotter(train_error, cv_error, self.directory, 'SVC_epsilon_analysis', xticks = xticks)