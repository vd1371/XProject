import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor



class Ensemble(Report):
    
    def __init__(self, df,
                        name = None,
                        split_size = 0.2,
                        should_shuffle = True,
                        num_top_features = 10):
        
        super(Ensemble, self).__init__(name)
        
        self.num_top_features = num_top_features
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.dates_train, self.dates_test = prepare_data_simple(df, split_size, should_shuffle= should_shuffle)
        self.X = pd.concat([self.X_train, self.X_test], axis = 0, join='outer')
        self.Y = pd.concat([self.Y_train, self.Y_test], axis = 0, join='outer')
    
    @timeit
    def run_ensemble_run(self, model_name = 'Ensemble'):
        reg1 = SVR(C=10, kernel= "rbf", epsilon = 0.1, gamma = 'auto')
        reg2 = KNeighborsRegressor(n_neighbors = 11)
        reg3 = RandomForestRegressor(n_estimators = 100)
        
        model = VotingRegressor([('RF', reg3)])
        model.fit(self.X_train, self.Y_train)
        
        self.evaluate_regression(self.Y_train, model.predict(self.X_train), self.dates_train, model_name+'-OnTrain', slicer = 1)
        self.evaluate_regression(self.Y_test, model.predict(self.X_test), self.dates_test, model_name+'-OnTest', slicer = 1)
        
def run():
    file_name = 'Steel_1877_2'
    myEns = Ensemble(file_name,file_name)
    myEns.run_ensemble_run()
    
    

if __name__ == "__main__":
    run()
        