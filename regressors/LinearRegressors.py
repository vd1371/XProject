import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel
from utils.SpecialPlotters import train_cv_analyzer_plotter
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

import statsmodels.api as sm
from scipy import stats
from sklearn.metrics.regression import mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge

class Linear(BaseModel):

    def __init__(self, name, dl):
        
        super().__init__(name, 'Linear', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        
        self.X, self.Y, _ = dl.load_all()

    @timeit
    def fit_ols(self):

        X_train, X_test = sm.add_constant(self.X_train), sm.add_constant(self.X_test)
        estimator = sm.OLS(self.Y_train, X_train)
        est = estimator.fit()
        self.log.info(str(est.summary()))

    @timeit
    def fit(self, model_name, alpha = 0.002, fit_intercept = True, should_analyze = False, start = 0.000001, end=2, step=2):

        if should_analyze and (model_name.lower() == 'lasso' or model_name.lower() == 'ridge'):
            
            lin_model = Lasso if model_name.lower() is 'lasso' else Ridge
            train_error_list, cv_error_list, xticks = [], [], []
        
            i = start
            while i < end:
                
                xticks.append(i)
                model = lin_model(alpha=i, fit_intercept = True, normalize = False, max_iter=10000)
                model.fit(self.X_train, self.Y_train)
                
                train_error_list.append(mean_squared_error(self.Y_train, model.predict(self.X_train)))
                cv_error_list.append(mean_squared_error(self.Y_test, model.predict(self.X_test)))
                
                print (f"Step {i:.4f} of regularization is done")
            
                i = i *step
        
            train_cv_analyzer_plotter(train_error_list, cv_error_list, self.directory, 'Lasso_Regularization_Analysis', xticks = xticks)
        
        else:

            if model_name.lower() is 'linear':
                model = LinearRegression(fit_intercept = True, normalize=False)
            else:
                lin_model = Lasso if model_name.lower() is 'lasso' else Ridge
                model = lin_model(alpha=alpha, fit_intercept = True, max_iter=10000)

            model.fit(self.X_train, self.Y_train)

            evaluate_regression(self.directory, self.X_train, self.Y_train,
                                model.predict(self.X_train), self.dates_train,
                                model_name+'-OnTrain', self.log, slicer = 1,
                                should_log_inverse = self.data_loader.should_log_inverse)
            
            evaluate_regression(self.directory, self.X_test, self.Y_test,
                                model.predict(self.X_test), self.dates_test,
                                model_name+'-OnTest', self.log, slicer = 1,
                                should_log_inverse = self.data_loader.should_log_inverse)

            joblib.dump(model, self.directory + f"/{model_name}.pkl")

            # Plotting the Importances
            coeffs = dict(zip(self.X_train.columns, model.coef_))
            report_feature_importance(self.directory, model.coef_, self.X_train.columns,
                                        self.n_top_features, model_name, self.log)

            self.log.info(f"{model_name} Coefficients:\n" + pprint.pformat(coeffs) + f" - Alpha value={round(alpha,4)}")