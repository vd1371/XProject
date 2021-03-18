import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel, R2
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

class Trees(BaseModel):
    
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
        
        self.n_estimators = params.pop('n_estimators', 100) 
        self.bootstrap = params.pop('bootstrap', 5)
        self.max_depth = params.pop('max_depth', 5)
        self.max_features = params.pop('max_features', 2)
        self.min_samples_leaf = params.pop('min_samples_leaf', 4)
        self.min_samples_split = params.pop('min_samples_split', 0.6)
        self.should_cross_val = params.pop('should_cross_val', True)
        self.n_jobs = params.pop('n_jobs', 1)
        self.verbose = params.pop('verbose', 1)

    def log_params(self):

        self.log.info(pprint.pformat({
            "Model_type": self.model_name,
            "n_estimators" : self.n_estimators,
            "bootstrap" : self.bootstrap,
            "max_depth" : self.max_depth,
            "max_features" : self.max_features,
            "min_samples_leaf" : self.min_samples_leaf,
            "min_samples_split" : self.min_samples_split,
            "should_cross_val" : self.should_cross_val,
            "n_jobs" : self.n_jobs,
            "verbose" : self.verbose,
            'random_state': self.dl.random_state
            }))
        
    @timeit
    def fit_random_forest(self):
        self.fit(RandomForestRegressor, 'RF')

    @timeit
    def fit_decision_tree(self):
        self.fit(DecisionTreeRegressor, 'DT')
    
    @timeit
    def fit_extra_trees(self):
        self.fit(ExtraTreesRegressor, 'ET')
    
    
    @timeit
    def fit(self, tree, model_name):

        self.initialize(model_name)
        self.load()
        self.log_params()
        
        if model_name != 'DT':
            model = tree(n_estimators=self.n_estimators, max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features, bootstrap=self.bootstrap,
                        n_jobs=self.n_jobs,  verbose=self.verbose, random_state = self.dl.random_state)
        else:
            model = tree( max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        
        model.fit(self.X_train, self.Y_train)
        print (f"{model_name} is fitted")
            
        if self.should_cross_val:
            r2_scorer = make_scorer(R2, greater_is_better=False)
            mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
            
            scores = cross_validate(model, self.X, self.Y, cv=self.k, verbose=0, scoring= {'MSE': mse_scorer, 'R2' : r2_scorer})
            self.log.info( f"Cross validation is done for {model_name}. RMSE: {(-np.mean(scores['test_MSE']))**0.5:.2f}, "
                                "MSE: {-np.mean(scores['test_MSE']):.2f} R2: {-np.mean(scores['test_R2']):.2f}")
        
            print (f"|- Cross validation is done for {model_name} "\
                            f"RMSE: {(-np.mean(scores['test_MSE']))**0.5:.2f},"\
                                f"MSE: {-np.mean(scores['test_MSE']):.2f}, "
                                    f"R2: {-np.mean(scores['test_R2']):.2f}-|")

        evaluate_regression(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = model,
                                model_name = model_name,
                                logger = self.log,
                                slicer = 1,
                                should_check_hetero = True,
                                should_log_inverse = False)

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
                search_models = RandomizedSearchCV(
                    estimator = tree,
                    param_distributions = grid,
                    n_iter = n_iter,
                    cv = self.k,
                    verbose=2,
                    n_jobs = self.n_jobs
                    )
            else:
                search_models = GridSearchCV(
                    estimator = tree,
                    param_distributions = grid,
                    cv = self.k,
                    verbose=2,
                    n_jobs = self.n_jobs
                    )
           
            search_models.fit(self.X, self.Y)
            
            self.log.info(f"\n\nBest params:\n{pprint.pformat(search_models.best_params_)}\n")
            self.log.info(f"\n\nBest score: {search_models.best_score_:0.4f}\n\n")
            print (search_models.best_score_)