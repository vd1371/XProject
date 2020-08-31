import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *


class Clustering(Report):
    
    def __init__(self, df, name = None):
        
        
        super(Clustering, self).__init__(name)

        self.X_train = open_csv(df)
        
        self.log.info('-------------- K-Means analysis on %s '%self.name)
        
    
    def findOptimalClusters(self,
                 min_num_clusters = 10,
                 max_num_clusters = 20,
                 n_init = 20,):
        
        train_err, test_err = [], []
        self.log.info(f"Trying to find the optimal number of clusters, min_clusters = {min_num_clusters}, max_clusters = {max_num_clusters}")
        
        for k in range (min_num_clusters, max_num_clusters+1):
            
            print ("-------------------------------------------------------------")
            print ("If number of clusters equals to %d, the fitting is started" %k)
            
            self.model = KMeans(n_clusters=k, n_init = n_init, verbose = 0, n_jobs = -1)
            self.model.fit(self.X_train)
            train_err.append(-self.model.score(self.X_train))
              
        
        xs = [ i+1 for i in range(min_num_clusters, max_num_clusters+1)]
        plt.clf()
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method to find the optimal K")
        plt.plot(xs, train_err, label = 'train_err')
        plt.grid(True, which="both")
        plt.legend()
        
        plt.savefig(self.directory + '/KMeans-' + self.name + '.png')
            
        plt.show()
    
    def fit(self,k=4):
        
        
        print ("Trying to fit the k-means clustering")
        self.model = KMeans(n_clusters=k, n_init = 20, verbose = 1, n_jobs = -1)
        self.model.fit(self.X_train)
        labels = self.model.predict(self.X_train)
        
        df = self.X_train.copy()
        
        df['K-Means'] = labels
        df.to_csv(self.directory + "/"+ self.name + '-KMeans.csv')
        
        plt.clf()
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)
        
        x = [i for i in range(len(self.X_train))]
        plt.step(x, labels, transform = rot + base)
        plt.show()
        plt.close()
        
    
    def predict(self, x = None , label = '-Kmeabs'):
        if isinstance(x, pd.DataFrame):
            labels = self.model.predict(x)
            PCA2D(x, labels, self.directory, label)
            PCA3D(x, labels, self.directory,label)
            return labels
        else:
            labels = self.model.predict(self.X_train)
            PCA2D(self.X_train, labels, self.directory, label)
            PCA3D(self.X_train, labels, self.directory, label)
            return
        
        
    
    
def run():
    
    myClustering = Clustering('Encoder-OnTrain', 'CPTTrain-Encoded')
#     myClustering.findOptimalClusters( min_num_clusters = 1, max_num_clusters=20)
    myClustering.fit(k=5)
    myClustering.predict(label='KMeans')

if __name__ == "__main__":
    run()        
        
