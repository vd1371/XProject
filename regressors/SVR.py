import numpy as np
import joblib, pprint
import matplotlib.pyplot as plt

from utils.BaseModel import BaseModel
from utils.SpecialPlotters import train_cv_analyzer_plotter
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

class SVMR(BaseModel):

    def __init__(self, dl):
        
        super().__init__('SVR', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        self.dl = dl
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        
        self.X, self.Y, _ = dl.load_all()

    def set_params(self, **params):
        
        self.kernel = params.get('kernel', 'linear') 
        self.gamma = params.get('gamma', 'auto')
        self.epsilon = params.get('epsilon', 0.1)
        self.C = params.get('C')

    def log_params(self):

        self.log.info(pprint.pformat({
            "Model_type": 'SVC',
            'kernel': self.kernel,
            'gamma': self.gamma,
            'C': self.C,
            'random_state': self.dl.random_state
            }))
    
    @timeit
    def fit_model(self):

        self.set_params()
        self.log_params()
        
        self.model = SVR(C = self.C,
                        kernel = self.kernel,
                        epsilon = self.epsilon,
                        gamma = self.gamma)
        self.model.fit(self.X_train, self.Y_train)
        
        if self.kernel == 'linear':
            report_feature_importance(self.directory, self.model.coef_[0], self.X_train, self.Y_train,
                                    self.n_top_features, 'SVR', self.log, False)

        evaluate_regression(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = self.model,
                                model_name = self.model_name,
                                logger = self.log,
                                slicer = 1,
                                should_check_hetero = True,
                                should_log_inverse = self.dl.should_log_inverse)

        joblib.dump(self.model, self.directory + f"/SVR.pkl")
    
    @timeit
    def regularization_analysis(self, start = 0.1, end = 1000, step = 2, kernel = 'rbf'):
        
        train_error, cv_error = [], []
        xticks = []
        
        i = start
        while i < end:
            
            xticks.append(i)
            model = SVR(C=i, kernel= kernel, epsilon = 0.1, gamma = 'auto')
            model.fit(self.X_train, self.Y_train)
            
            train_error.append(mean_squared_error(self.Y_train, model.predict(self.X_train)))
            cv_error.append(mean_squared_error(self.Y_test, model.predict(self.X_test)))
            
            print (f"Step {i} of SVM regularization is done")
        
            i = i *step
        
        train_cv_analyzer_plotter(train_error, cv_error, self.directory, 'SVM_regularization_analysis', xticks = xticks)
    
    @timeit
    def epsilon_analysis(self, C=25, start = 0.00001, end = 100, step = 2):
        
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

        train_cv_analyzer_plotter(train_error, cv_error, self.directory, 'SVR_epsilon_analysis', xticks = xticks)