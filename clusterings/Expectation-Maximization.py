import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

from sklearn.mixture import GaussianMixture


class Clustering(Report):
    
    def __init__(self, df, name= None,
                 slicer = 0.2):
        
        
        super(Clustering, self).__init__(name)
        self.slicer = slicer
        
        self.X_train = open_csv(df)
        
        self.log.info('-------------- Expectation Maximization analysis on %s '%self.name)
    
    def findOptimalComponents(self,
                 min_num_clusters = 10,
                 max_num_clusters = 20,
                 n_init = 20):
        
        train_err, test_err = [], []
        
        self.log.info(f"Trying to find the optimal number of clusters, min_clusters = {min_num_clusters}, max_clusters = {max_num_clusters}")
        
        for k in range (min_num_clusters, max_num_clusters+1):
            
            print ("-------------------------------------------------------------")
            print ("If number of clusters equals to %d, the fitting is started" %k)
            
            model = GaussianMixture(n_components = k, n_init = n_init, verbose = 0)
            model.fit(self.X_train)
            train_err.append(-model.score(self.X_train))
              
        
        xs = [ i+1 for i in range(min_num_clusters, max_num_clusters+1)]
        plt.clf()
        plt.xlabel("Number of Components - EXMA")
        plt.ylabel("Inertia")
        plt.title("Elbow Method to find the optimal K")
        plt.plot(xs, train_err, label = 'train_err')
        plt.grid(True, which="both")
        plt.legend()
        
        plt.savefig(self.directory + '/EXMA-' + self.name + '.png')
            
        plt.show()
        
    
    def run(self, n_components=2, max_iter = 100, n_init = 1, verbose = 1):
        
       
        labels = [1 for i in range(len(self.X_train))]
        PCA2D(self.X_train, labels, self.directory, 'PureData')
        PCA3D(self.X_train, labels, self.directory, 'PureData')
        
        
        model = GaussianMixture(n_components=n_components, max_iter = max_iter, n_init = n_init, verbose = verbose)
        model.fit(self.X_train)
        labels = model.predict(self.X_train)
        
        proba = model.predict_proba(self.X_train)
        
        print("Model Score:", model.score(self.X_train))
        
        PCA2D(self.X_train, labels, self.directory, 'ExMa')
        PCA3D(self.X_train, labels, self.directory, 'ExMa')
        
        
        self.X_train['EXMA'] = labels
        self.X_train.to_csv(self.directory + "/"+ self.name + '-EXMA.csv')
        
        plt.clf()
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)
        
        x = [i for i in range(len(self.X_train))]
        plt.step(x, labels, transform = rot + base)
        plt.show()
        plt.close()
        
    
    
def run():
    
    myClustering = Clustering('Encoder-OnTrain', 'CPTTrain-Encoded', slicer = 0.2)
#     myClustering.findOptimalComponents( min_num_clusters = 1, max_num_clusters=10)
    myClustering.run(n_components=8, max_iter = 100, n_init = 1, verbose = 1)

if __name__ == "__main__":
    run()        
        
