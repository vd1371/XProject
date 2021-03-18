#Loading dependencies
import numpy as np
import joblib
import pprint

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.FeatureImportanceReport import report_feature_importance
from utils.ClassificationReport import evaluate_classification
from utils.FeatureImportanceReport import report_feature_importance

from sklearn.neighbors import KNeighborsClassifier


class KNNC(BaseModel):
    
    def __init__(self, name, dl):
        
        super().__init__(name, 'KNNC', dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        self.dl = dl
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, \
                self.dates_train, self.dates_test = dl.load_with_test()
    
    @timeit
    def fit(self, n = 5):

        self.log.info(f'KNN Classifier is about to be fit on {self.name} with n = {n}')
        model = KNeighborsClassifier(n_neighbors=n, n_jobs = -1)
        model.fit(self.X_train, self.Y_train)
        
        evaluate_classification(['OnTrain', self.X_train, self.Y_train, self.dates_train],
                                ['OnTest', self.X_test, self.Y_test, self.dates_test],
                                direc = self.directory,
                                model = model,
                                model_name = model_name,
                                logger = self.log,
                                slicer = 1)
    
    @timeit
    def neighbour_analysis(self, start = 1, end = 20, step = 1):
        
        train_error_list = []
        cv_error_list = []
        x = []
        
        labels = []
        i = start
        while i < end:
            
            labels.append(i)
            
            model = KNeighborsClassifier(n_neighbors = i, n_jobs = -1)
            model.fit(self.X_train, self.Y_train)
            
            train_err, cv_err = 1-model.score(self.X_train, self.Y_train), 1-model.score(self.X_test, self.Y_test)
            
            train_error_list.append(train_err)
            cv_error_list.append(cv_err)
            
            print (f"KKN with {i} neighbours - train_err={train_err:.4f} - cv_err={cv_err:.4f}")
        
            i = i + step
        
        x = [i+1 for i in range(len(train_error_list))]
        plt.clf()
        plt.xlabel('Steps')
        plt.ylabel('Regression Error: MSE')
        plt.title(f'Regularization Analysis-{start}:{end}-step:{step}')
        plt.plot(x, train_error_list, label = 'Train_error')
        plt.plot(x, cv_error_list, label = 'Test Error')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.directory + '/KNNR_Neighbours.png')
        plt.show()
        plt.close()
        
                    
           
def run():
    myKNN = KNNC('Digit_train', 'Digit_train')
    myKNN.fit(n = 11)
#     mySVM.regularization_analysis(start=0.1, end=10, step=1.2)

if __name__ == "__main__":
    run()

        