import numpy as np
import joblib
import pprint
import utils


from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import make_scorer

from ._LinearRegressorUtils import *

class Linear(utils.BaseModel):

    def __init__(self, dl):
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        self.dl = dl

    def initialize(self, model_name):
        super().__init__(model_name, self.dl)

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

    @utils.timeit
    def fit(self, model_name = 'linear'):

        self.initialize(model_name)
        self.load()
        self.log_params()

        if model_name.lower() == 'ols':
            fit_and_log_ols(**self.__dict__)

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
                cross_validate_model(model, model_name, **self.__dict__)

            model.fit(self.X_train, self.Y_train)

            utils.evaluate_regression(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = model,
                                model_name = model_name,
                                logger = self.log,
                                slicer = 1,
                                should_check_hetero = True,
                                should_log_inverse = self.dl.should_log_inverse)

            joblib.dump(model, self.directory + f"/{model_name}.pkl")

            utils.report_feature_importance(self.directory, model.coef_, self.X, self.Y,
                                    self.n_top_features, model_name, self.log)


    @utils.timeit
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
    
        utils.train_cv_analyzer_plotter(train_error_list, cv_error_list, self.directory,
                                    f'{model_name}_Regularization_Analysis', xticks = xticks)