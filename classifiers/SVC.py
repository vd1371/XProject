import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

from sklearn.svm import SVC
 
class SVMC(Report):
    
    def __init__(self, df,
                 name = None,
                 should_shuffle = True,
                 split_size = 0.4,
                 is_imbalanced = False):
        
        super(SVMC, self).__init__(name, 'SVC')
        
        self.X_train,self.X_cv,self.X_test,self.Y_train,self.Y_cv,self.Y_test,self.dates_train,self.dates_cv,self.dates_test = prepare_data_simple(df,
                                                                                                                                                    split_size,
                                                                                                                                                    is_cv = True,
                                                                                                                                                    should_shuffle = should_shuffle,
                                                                                                                                                    is_imbalanced = is_imbalanced)
        
        self.log.info('-------------- SVC is about to be fit on %s '%self.name)
    
    @timeit
    def fit(self, C=1):
        
        model = SVC(C=1, kernel= "rbf", gamma = 'auto')
        model.fit(self.X_train, self.Y_train)
        
        self.evaluate_classification(self.Y_train, model.predict(self.X_train), self.dates_train, 'SCV-OnTrain')
        self.evaluate_classification(self.Y_test, model.predict(self.X_test), self.dates_test, 'SVC-OnTest')
    
    @timeit
    def regularization_analysis(self, start = 1, end = 100000, step = 2):
        
        train_error = []
        cv_error = []
        x = []
         
        i = start
        while i < end:
            
            
            model = SVC(C=i, kernel= "rbf", gamma = 'auto')
            model.fit(self.X_train, self.Y_train)
            
            train_err, cv_err = 1-model.score(self.X_train, self.Y_train), 1-model.score(self.X_cv, self.Y_cv)
            
            train_error.append(train_err)
            cv_error.append(cv_err)
            
            print (f"Step {i:.2} of SVM regularization - train_err={train_err:.4f} - cv_err={cv_err:.4f}")
            
            x.append(i)
            i = i *step
        
        plt.clf()
        plt.xlabel('Steps')
        plt.ylabel('Classification Error')
        plt.title('Regularization Analysis')
        plt.plot(x, train_error, label = 'Train_error')
        plt.plot(x, cv_error, label = 'CV Error')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.directory + '/SVC_Regularization Analysis.png')
        plt.show()
        plt.close()
        
        
           
def run():
    mySVM = SVMC('MisCondV', 'MisCondV', 
                 should_shuffle = True,
                 split_size = 0.3,
                 is_imbalanced = True)
    mySVM.fit(C = 1)
#     mySVM.regularization_analysis(start=0.01, end=10000, step=2)
    
    

if __name__ == "__main__":
    run()

        