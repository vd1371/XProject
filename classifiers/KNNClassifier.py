import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

from sklearn.neighbors import KNeighborsClassifier


class KNNC(Report):
    
    def __init__(self, df,
                 name = None,
                 should_shuffle = True,
                 split_size = 0.2):
        
        super(KNNC, self).__init__(name, 'KNN')
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.dates_train, self.dates_test = prepare_data_simple(df, split_size, should_shuffle= should_shuffle)
        self.log.info('-------------- KNN Classifier is about to be fit on %s '%self.name)
    
    @timeit
    def fit(self, n = 5):
        
        model = KNeighborsClassifier(n_neighbors=n, n_jobs = -1)
        model.fit(self.X_train, self.Y_train)
        
        self.evaluate_classification(self.Y_train, model.predict(self.X_train), self.dates_train, 'KNNC-OnTrain')
        self.evaluate_classification(self.Y_test, model.predict(self.X_test), self.dates_test, 'KNNC-OnTest')
    
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

        