import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)

from Reporter import *

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV 

class Linear(Report):
    
    def __init__(self, df ,
                        name = None,
                        split_size = 0.2,
                        should_shuffle = True,
                        num_top_features = 10,
                        is_imbalanced = False):
        
        super(Linear, self).__init__(name, 'LogisticRegression')
        
        self.num_top_features = num_top_features
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.dates_train, self.dates_test = prepare_data_simple(df, split_size, False, should_shuffle,
                                                                                                                      is_imbalanced = is_imbalanced)
        self.X = pd.concat([self.X_train, self.X_test], axis = 0, join='outer')
        self.Y = pd.concat([self.Y_train, self.Y_test], axis = 0, join='outer')
        
        
    @timeit
    def logistic_regression(self):
        
        model = LogisticRegressionCV(Cs = 10, fit_intercept = True, penalty= 'l1', solver = 'saga')
        model.fit(self.X_train, self.Y_train)
        
        coeffs = {}
        for i in range (len(model.coef_[0])):
            coeffs[self.X.columns[i]] = model.coef_[0][i]
        coeffs['c'] = model.intercept_[0]
        
        self.evaluate_classification(self.Y_train, model.predict(self.X_train), self.dates_train, 'LogisticRegression-OnTrain')
        self.evaluate_classification(self.Y_test, model.predict(self.X_test), self.dates_test,'LogisticRegression-OnTest')
        
        # Plotting the Importances
        feature_importances_ = {}
        for i in range(len(model.feature_importances_)):
            feature_importances_[self.X_train.columns[i]] = model.feature_importances_[i]
        
        self.report_feature_importance(feature_importances_, self.num_top_features, label = model_name)
        
    @timeit
    def ridge (self, alphas = [0.1, 10, 100]):
        
        model = RidgeClassifierCV(alphas = alphas, fit_intercept=True, normalize=False)
        model.fit(self.X_train, self.Y_train)
        coeffs = {}
        for i in range (len(model.coef_[0])):
            coeffs[self.X.columns[i]] = model.coef_[0][i]
        coeffs['c'] = model.intercept_[0]

        
        report_feature_importance(self.log, self.directory, coeffs, self.num_top_features, label = "Ridge Classifier")
        
        self.evaluate_classification(self.Y_train, model.predict(self.X_train), self.dates_train, 'RidgeLogisticRegression-OnTrain')
        self.evaluate_classification(self.Y_test, model.predict(self.X_test), self.dates_test,'RidgeLogisticRegression-OnTest')
        
@timeit  
def run():
    
    lin = Linear("MisCond", 'MisCond', should_shuffle= True, num_top_features = 10, is_imbalanced = True)
    lin.logistic_regression()
    lin.ridge([1e-3, 1e-2, 1e-1, 1, 10])
    
    
if __name__ == "__main__":
    run()    
    