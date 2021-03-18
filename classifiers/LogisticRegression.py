#Loading dependencies
import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.FeatureImportanceReport import report_feature_importance
from utils.ClassificationReport import evaluate_classification
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.linear_model import LogisticRegression, RidgeClassifier

class Logit(BaseModel):
    
    def __init__(self, name, dl):
        
        super().__init__(name, 'Logit', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        self.dl = dl
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
        
        self.X, self.Y, _ = dl.load_all()
        
        
    @timeit
    def fit(self):
        
        model = LogisticRegression(C = 1, fit_intercept = True, penalty= 'l2', solver = 'lbfgs')
        model.fit(self.X_train, self.Y_train)

        coeffs = {}
        for i in range (len(model.coef_[0])):
            coeffs[self.X.columns[i]] = model.coef_[0][i]
        coeffs['c'] = model.intercept_[0]

        evaluate_classification(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = model,
                                model_name = self.model_name,
                                logger = self.log,
                                slicer = 1)
        
        joblib.dump(model, self.directory + f"/Logit.pkl")

        # Plotting the Importances
        report_feature_importance(self.directory, model.coef_[0], self.X, self.Y,
                                    self.n_top_features, "Logit", self.log)