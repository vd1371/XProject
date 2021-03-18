#Loading dependencies
import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.FeatureImportanceReport import report_feature_importance
from utils.ClassificationReport import evaluate_classification
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import catboost as ctb

class Boosting(BaseModel):

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
        
        self.n_estimators = params.pop("n_estimators", 100)
        self.learning_rate = params.pop("learning_rate", 0.1)
        self.max_depth = params.pop("max_depth", 12)
        self.n_jobs = params.pop("n_jobs", -1)
        self.verbose = params.pop("verbose", 1)
        self.objective = params.pop("objective", 'multi:softmax')
        # self.objective = params.pop("objective", 'binary:logistic')
        self.l1 = params.pop("reg_alpha", 0)
        self.l2 = params.pop('reg_lambda', 0.000001)
        self.n_iter = params.pop("n_iter", 200)

    def log_params(self):

        self.log.info(pprint.pformat({
            "Model_type": self.model_name,
            "learning_rate": self.learning_rate,
            "objective of XGBOOST": self.objective,
            "n_estimators" : self.n_estimators,
            "max_depth" : self.max_depth,
            "n_jobs" : self.n_jobs,
            "verbose" : self.verbose,
            "L1" : self.l1,
            "L2" : self.l2,
            "n_iter for catboost" : self.n_iter,
            'random_state': self.dl.random_state
            }))

    
    @timeit
    def xgb_run(self):

        metrics = ['merror', 'mlogloss']
        # metrics = ['error', 'logloss']

        self.initialize("XGBOOST")
        self.load()
        self.log_params()
        
        model = xgb.XGBClassifier(max_depth=self.max_depth,
                                    learning_rate=self.learning_rate,
                                    n_estimators=self.n_estimators, 
                                    verbosity=self.verbose,
                                    objective=self.objective,
                                    reg_alpha=self.l1,
                                    reg_lambda= self.l2,
                                    booster='gbtree',
                                    n_jobs=self.n_jobs,
                                    random_state = self.dl.random_state)

        eval_set = [(self.X_train, self.Y_train), (self.X_test, self.Y_test)]
        model.fit(self.X_train, self.Y_train, eval_metric=metrics, eval_set=eval_set, verbose=self.verbose)

        feature_importances_ = model.get_booster().get_score(importance_type="gain")
        feature_importances_ = np.array(list(feature_importances_.values()))

        self._get_report(model, 'XGB', model.feature_importances_)
    
    @timeit
    def catboost_run(self, warm_up = False):

        model_name = 'CATBOOST'

        self.initialize(model_name)
        self.load()
        self.log_params()

        if not warm_up:
            model = ctb.CatBoostClassifier(iterations = self.n_iter,
                                            learning_rate = self.learning_rate,
                                            depth = self.max_depth,
                                            l2_leaf_reg = self.l2,
                                            allow_writing_files=False,
                                            eval_metric = 'Accuracy')

            eval_set = ctb.Pool(self.X_test, self.Y_test)

            model.fit(self.X_train, self.Y_train, eval_set = eval_set, plot=True)

            model.save_model(self.directory + f"/{model_name}")

        else:
            model = ctb.CatBoostClassifier()
            model.load_model(self.directory + f"/{model_name}")


        self._get_report(model, model_name, model.feature_importances_)
        

    def _get_report(self, model, model_name, feature_importances_):

        evaluate_classification(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = model,
                                model_name = model_name,
                                logger = self.log,
                                slicer = 1)

        report_feature_importance(self.directory,
                                feature_importances_,
                                self.X,
                                self.Y,
                                self.n_top_features,
                                model_name,
                                self.log,
                                should_plot_heatmap = True)