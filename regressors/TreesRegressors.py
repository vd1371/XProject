import numpy as np
import joblib

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.metrics.regression import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

class Trees(BaseModel):
    
    def __init__(self, name, dl):
        
        super().__init__(name, 'Trees', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        
        self.X, self.Y, _ = dl.load_all()
        
    def set_params(self, n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                  max_features='auto', bootstrap=True, n_jobs=-1,  verbose=1, should_cross_val = True):
    
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.should_cross_val = should_cross_val
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def set_trees(self, trees_dict):
        self.trees = trees_dict
    
    @timeit
    def fit(self, tree, model_name):
        
        self.log.info(f'---{model_name} with {self.n_estimators} estimators is about to fit')
        if model_name != 'DT':
            model = tree(n_estimators=self.n_estimators, max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features, bootstrap=self.bootstrap,
                        n_jobs=self.n_jobs,  verbose=self.verbose)
        else:
            model = tree( max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        
        model.fit(self.X_train, self.Y_train)
        print (f"{model_name} is fitted")
            
        if self.should_cross_val:
            r2_scorer = make_scorer(R2, greater_is_better=False)
            mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
            
            scores = cross_validate(model, self.X, self.Y, cv=self.k, verbose=0, scoring= {'MSE': mse_scorer, 'R2' : r2_scorer})
            self.log.info( f"Cross validation is done for {model_name}. RMSE: {(-np.mean(scores['test_MSE']))**0.5:.2f}, MSE: {-np.mean(scores['test_MSE']):.2f} R2: {-np.mean(scores['test_R2']):.2f}")
        
            print (f"Cross validation is done for {model_name}. RMSE: {(-np.mean(scores['test_MSE']))**0.5:.2f},  MSE: {-np.mean(scores['test_MSE']):.2f} R2: {-np.mean(scores['test_R2']):.2f}")
        
        evaluate_regression(self.directory, self.X_train,
                            self.Y_train, model.predict(self.X_train),
                            self.dates_train, model_name+'-OnTrain',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)
        evaluate_regression(self.directory, self.X_test,
                            self.Y_test, model.predict(self.X_test),
                            self.dates_test, model_name+'-OnTest',
                            self.log, slicer = 1,
                            should_log_inverse = self.data_loader.should_log_inverse)

        joblib.dump(model, self.directory + f"/{model_name}.pkl")
        
        # Plotting the Importances
        report_feature_importance(self.directory, model.feature_importances_, self.X_train.columns,
                                    self.n_top_features, model_name, self.log)
        
    @timeit
    def tune_trees(self, grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                'max_features': ['auto', 'sqrt'],
                                'max_depth': [int(x) for x in np.linspace(5, 20, num = 15)] + [None],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'bootstrap': [True, False]},
                                should_random_search = False,
                                n_iter = 1):
        
        for model_name in self.trees:
            
            self.log.info(f'------ {model_name} is going to be Tunned with \n -----')
            tree = self.trees[model_name]()
            if should_random_search:
                search_models = RandomizedSearchCV(estimator = tree, param_distributions = grid, n_iter = n_iter, cv = self.k, verbose=2, n_jobs = self.n_jobs)
            else:
                search_models = GridSearchCV(estimator = tree, param_distributions = grid, cv = self.k, verbose=2, n_jobs = self.n_jobs)
           
            search_models.fit(self.X, self.Y)
            
            self.log.info(f"\n\nBest params:\n{pprint.pformat(search_models.best_params_)}\n")
            self.log.info(f"\n\nBest score: {search_models.best_score_:0.4f}\n\n")
            print (search_models.best_score_)
            
    def fit_all(self):
        
        for model_name in self.trees:
            print (f"{model_name} is about to start")
            tree = self.trees[model_name]
            self.fit(tree, model_name)