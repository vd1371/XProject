import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel, R2
from utils.SpecialPlotters import train_cv_analyzer_plotter
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import make_scorer

class Linear(BaseModel):

    def __init__(self, name, dl):
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        self.dl = dl
        self.name = name

    def initialize(self, model_name):
        super().__init__(self.name, model_name, self.dl)

    def load(self):
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = self.dl.load_with_test()
        
        self.X, self.Y, _ = self.dl.load_all()

    def set_params(self, **params):
        
        self.alpha = params.pop('alpha', 0.002) 
        self.fit_intercept = params.pop('fit_intercept', True)
        self.should_cross_val = params.pop('should_cross_val', False)

    def log_params(self):

        self.log.info(pprint.pformat({
            "Model_type": self.model_name,
            "alpha" : self.alpha,
            "fit_intercept" : self.fit_intercept,
            "should_cross_val" : self.should_cross_val,
            'random_state': self.dl.random_state
            }))

    def fit_ols(self):
        self.fit(model_name = 'OLS')

    def fit_linear(self):
        self.fit(model_name = 'Linear')

    def fit_lasso(self):
        self.fit(model_name = 'Lasso')

    def fit_ridge(self):
        self.fit(model_name = 'Ridge')

    @timeit
    def fit(self, model_name = 'linear'):

        self.initialize(model_name)
        self.load()
        self.log_params()

        if model_name.lower() == 'ols':
            X_train, X_test = sm.add_constant(self.X_train), sm.add_constant(self.X_test)
            model = sm.OLS(self.Y_train, X_train)
            model = model.fit()
            self.log.info(str(model.summary()))

        else:

            if model_name.lower() == 'linear':
                model = LinearRegression(fit_intercept = self.fit_intercept,
                                        normalize=False)
            else:
                lin_model = Lasso if model_name.lower() is 'lasso' else Ridge
                model = lin_model(alpha = self.alpha,
                                    fit_intercept = self.fit_intercept,
                                    max_iter = 10000)

            if self.should_cross_val:
                r2_scorer = make_scorer(R2, greater_is_better=False)
                mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
                
                scores = cross_validate(model, self.X, self.Y, 
                                        cv=self.k, verbose=0, scoring= {'MSE': mse_scorer, 'R2' : r2_scorer})
                
                self.log.info( f"Cross validation is done for {model_name}. "\
                                    f"RMSE: {(-np.mean(scores['test_MSE']))**0.5:.2f}, "\
                                        f"MSE: {-np.mean(scores['test_MSE']):.2f},"\
                                            f" R2: {-np.mean(scores['test_R2']):.2f}")
            
                print (f"|- Cross validation is done for {model_name} "\
                            f"RMSE: {(-np.mean(scores['test_MSE']))**0.5:.2f},"\
                                f"MSE: {-np.mean(scores['test_MSE']):.2f}, "
                                    f"R2: {-np.mean(scores['test_R2']):.2f} -|")

            model.fit(self.X_train, self.Y_train)

            evaluate_regression(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = model,
                                model_name = model_name,
                                logger = self.log,
                                slicer = 1,
                                should_check_hetero = True,
                                should_log_inverse = self.dl.should_log_inverse)

            joblib.dump(model, self.directory + f"/{model_name}.pkl")

            # Plotting the Importances
            coeffs = dict(zip(self.X_train.columns, model.coef_))
            report_feature_importance(self.directory, model.coef_, self.X, self.Y,
                                    self.n_top_features, model_name, self.log)

            self.log.info(f"{model_name} Coefficients:\n" + pprint.pformat(coeffs))

    @timeit
    def analyze(self, model_name = 'Lasso', start = 0.0000001, end=100, step=2):

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
    
        train_cv_analyzer_plotter(train_error_list, cv_error_list, self.directory,
                                    f'{model_name}_Regularization_Analysis', xticks = xticks)