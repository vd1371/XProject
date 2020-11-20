import numpy as np
import joblib
import pprint
import time

from utils.BaseModel import BaseModel
from utils.SpecialPlotters import train_cv_analyzer_plotter
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.metrics.regression import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

class KNNR(BaseModel):

    def __init__(self, name, dl):
        
        super().__init__(name, 'KNN', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        
        self.X, self.Y, _ = dl.load_all()
    
    @timeit
    def fit_model(self, n = 5 ):

        self.log.info(pprint.pformat({
            "n_neighbours": n
            }))
        
        model = KNeighborsRegressor(n_neighbors = n, n_jobs = -1)
        model.fit(self.X_train, self.Y_train)

        evaluate_regression(self.directory, self.X_train,
                            self.Y_train, model.predict(self.X_train),
                            self.dates_train, 'KNN-OnTrain',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)
        evaluate_regression(self.directory, self.X_test,
                            self.Y_test, model.predict(self.X_test),
                            self.dates_test, 'KNN-OnTest',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)

        joblib.dump(model, self.directory + f"/KNN.pkl")

    
    @timeit
    def neighbour_analysis_(self, start = 1, end = 20, step = 1):
        # There is somethng wrong with the predict method of KNN, maybe it will be solved in the future
        train_error, cv_error = [], []
        xticks = []

        for i in range(start, end, step):
            
            xticks.append(i)
            
            model = KNeighborsRegressor(n_neighbors = i, n_jobs = -1)
            model.fit(self.X_train, self.Y_train)

            model.predict(self.X_train)
            
            # train_error.append(mean_squared_error(self.Y_train, model.predict(self.X_train)))
            # cv_error.append(mean_squared_error(self.Y_test, model.predict(self.X_test)))

            del model

            print ("we are sleeping")
            time.sleep(5)
            
            print (f"KKN with {i} neighbours is analyzed")

        # train_cv_analyzer_plotter(train_error, cv_error, self.directory, 'knn_neighbour_analysis', xticks = xticks)