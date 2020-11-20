import os, pprint
import numpy as np
import joblib

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

        
class Boosting_XGB(BaseModel):
    
    def __init__(self, name, dl):
                        
        super().__init__(name, 'XGB', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        #splitting data into X and Y
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        self.X, self.Y, _ = dl.load_all()
        
        
    def setParams(self, n_estimators = 2000, learning_rate = 0.05, max_depth = 5,
                    max_features = 2, min_samples_leaf=4, min_samples_split = 0.6,
                    reg_alpha=0.0004, should_cross_val=True, n_jobs = 1,
                    verbose = 0):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.should_cross_val = should_cross_val
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.reg_alpha = reg_alpha

        self.log.info(pprint.pformat({
            "n_estimators" : n_estimators,
            "learning_rate" : learning_rate,
            "max_depth" : max_depth,
            "max_features" : max_features,
            "min_samples_leaf" : min_samples_leaf,
            "min_samples_split" : min_samples_split,
            "should_cross_val" : should_cross_val,
            "n_jobs" : n_jobs,
            "verbose" : verbose,
            "reg_alpha" : reg_alpha,
            }))
    
    @timeit
    def xgb_run(self):
        
        data_matrix = xgb.DMatrix(data=self.X_train, label=self.Y_train)
        
        model = xgb.XGBRegressor(max_depth=self.max_depth, 
                                 learning_rate=self.learning_rate,
                                 n_estimators=self.n_estimators, 
                                 verbosity=self.verbose,
                                 reg_alpha=self.reg_alpha,
                                 booster='gbtree',
                                 n_jobs=self.n_jobs)
        eval_set = [(self.X_train, self.Y_train), (self.X_test, self.Y_test)]
        model.fit(self.X_train, self.Y_train, eval_set=eval_set, verbose=True)
        
        if self.should_cross_val:
            scores = cross_val_score(model, self.X, self.Y, cv=self.k, verbose=0)
            self.log.info(f"---- Cross validation with {self.k} groups----\n\nThe results on each split" +\
                                 str(scores)+"\n")
            self.log.info(f"The average of the cross validation is {scores.mean()}\n")
        print (f"Cross validation is done for {self.name}")
        
        joblib.dump(model, self.directory + f'/{self.name}.pkl', compress=3)
        
        evaluate_regression(self.directory, self.X_train,
                            self.Y_train, model.predict(self.X_train),
                            self.dates_train, 'XGB-OnTrain',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)
        evaluate_regression(self.directory, self.X_test,
                            self.Y_test, model.predict(self.X_test),
                            self.dates_test, 'XGB-OnTest',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)
    
    def tune(self, grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                'learning_rate': np.linspace(0.01, 0.2, 10),
                                'max_features': ['auto', 'sqrt'],
                                'max_depth': [int(x) for x in np.linspace(5, 20, num = 15)] + [None],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'reg_alpha': np.linspace(0.0001, 0.02, 200)},
                                should_random_search = False,
                                n_iter = 1):
    
            
        self.log.info(f'------ {self.name} is going to be Tunned with \n -----')
        model = xgb.XGBRegressor()
        if should_random_search:
            search_models = RandomizedSearchCV(estimator = model,
                                                param_distributions = grid,
                                                n_iter = n_iter,
                                                cv = self.k,
                                                verbose=2,
                                                n_jobs = -1)
        else:
            search_models = GridSearchCV(estimator = model,
                                        param_distributions = grid,
                                        cv = self.k,
                                        verbose=2,
                                        n_jobs = -1)
       
        search_models.fit(self.X, self.Y)
        
        self.log.info(f"\n\nBest params:\n{pprint.pformat(search_models.best_params_)}\n")
        self.log.info(f"\n\nBest score: {search_models.best_score_:0.4f}\n\n")
        print (search_models.best_score_)
    
    def load_xgb(self):
        model = joblib.load(self.directory + f'/XGBOOST-{self.name}.pkl')
        input_params = self.X.as_matrix()
        pre = model.predict(input_params)
        print (pre)
        
@timeit
def run():
    file = 'HACKA1'
    
    bst = Boosting(file, name = file, split_size=0.2, should_shuffle=True, k=5, num_top_features = 10)
    bst.setParams(n_estimators = 100,
                  learning_rate = 0.02,
                  max_depth = 5,
                  max_features = 'auto',
                  min_samples_leaf= 1,
                  min_samples_split = 2,
                  reg_alpha=0.005,
                  should_cross_val=False, n_jobs = -1, verbose = 1)
    
#     bst.tune( grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 20)],
#                     'learning_rate': np.linspace(0.01, 0.2, 10),
#                     'max_features': ['auto', 'sqrt'],
#                     'max_depth': [int(x) for x in np.linspace(5, 100, num = 15)],
#                     'min_samples_split': np.arange(2,20,4),
#                     'min_samples_leaf': np.arange(2,20,4),
#                     'reg_alpha': np.linspace(0.0001, 0.02, 200)},
                    
#                     should_random_search = True,
#                     n_iter = 200)
    bst.xgb_run()
#     bst.load_xgb()

if __name__ == "__main__":
    run()

        