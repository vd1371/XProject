#Loading dependencies
import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.FeatureImportanceReport import report_feature_importance
from utils.ClassificationReport import evaluate_classification
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


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
        self.fit(RandomForestClassifier, 'RF')

    @timeit
    def fit_decision_tree(self):
        self.fit(DecisionTreeClassifier, 'DT')
    
    @timeit
    def fit_extra_trees(self):
        self.fit(ExtraTreesClassifier, 'ET')

    @timeit
    def fit_balanced_random_forest(self):
        self.fit(BalancedRandomForestClassifier, 'BRF')
    
    @timeit
    def fit(self, tree, model_name):

        self.initialize(model_name)
        self.load()
        self.log_params()
        
        if model_name != 'DT':
            model = tree(n_estimators=self.n_estimators, max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features, bootstrap=self.bootstrap,
                        n_jobs=self.n_jobs,  verbose=self.verbose)
        else:
            model = tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        
        model.fit(self.X_train.values, self.Y_train.values)
        print (f"{model_name} is fitted")
        
        if self.should_cross_val:
            scores = cross_val_score(model, self.X, self.Y, cv=self.k, verbose=0)
            self.log.info(f"---- Cross validation with {self.k} groups----\n\nThe results on each split" + str(scores)+"\n")
            self.log.info(f"The average of the cross validation is {np.mean(scores):.2f}\n")
            
            print (f"|- Cross validation is done for {model_name}. Accuracy: {np.mean(scores):.2f} -|")
        
        evaluate_classification(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = model,
                                model_name = model_name,
                                logger = self.log,
                                slicer = 1)

        joblib.dump(model, self.directory + f"/{model_name}.pkl")
        
        # Plotting the Importances
        report_feature_importance(self.directory, model.feature_importances_, self.X, self.Y,
                                    self.n_top_features, model_name, self.log)

    
    def tune_trees(self, model, model_name,
                        grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                'max_features': ['auto', 'sqrt'],
                                'max_depth': [int(x) for x in np.linspace(5, 20, num = 15)] + [None],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'bootstrap': [True, False]},
                                should_random_search = False,
                                n_iter = 1):
            
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