import numpy as np
import joblib, pprint
import matplotlib.pyplot as plt

from utils.BaseModel import BaseModel
from utils.SpecialPlotters import train_cv_analyzer_plotter
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.metrics.regression import mean_squared_error
from sklearn.svm import SVR

class SVMR(BaseModel):

    def __init__(self, name, dl):
        
        super().__init__(name, 'SVM', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        
        self.X_train, self.X_cv, self.X_test, \
            self.Y_train, self.Y_cv, self.Y_test, \
            self.dates_train, self.dates_cv, self.dates_test = dl.load_with_csv()
        
        self.X, self.Y, _ = dl.load_all()
    
    @timeit
    def fit_model(self, C = 1, kernel = 'rbf', epsilon = 0.3, gamma = 'auto' ):

        self.log.info(pprint.pformat({
            "C (regularization)": C,
            "kernel": kernel,
            "epsilon": epsilon,
            "gamma": gamma
            }))
        
        self.model = SVR(C = C, kernel = kernel, epsilon=epsilon, gamma = gamma)
        self.model.fit(self.X_train, self.Y_train)
        
        if kernel == 'linear':
            report_feature_importance(self.directory, self.model.coef_[0], self.X_train.columns,
                                    self.n_top_features, 'SVR', self.log)

        evaluate_regression(self.directory, self.X_train,
                            self.Y_train, self.model.predict(self.X_train),
                            self.dates_train, 'SVR-OnTrain',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)
        evaluate_regression(self.directory, self.X_test,
                            self.Y_test, self.model.predict(self.X_test),
                            self.dates_test, 'SVR-OnTest',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)

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
            cv_error.append(mean_squared_error(self.Y_cv, model.predict(self.X_cv)))
            
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
            cv_error.append(mean_squared_error(self.Y_cv, model.predict(self.X_cv)))
            
            print (f"Step {i} of SVM epsilon_analysis is done")
        
            i = i *step

        train_cv_analyzer_plotter(train_error, cv_error, self.directory, 'SVR_epsilon_analysis', xticks = xticks)