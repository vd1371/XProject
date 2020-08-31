import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, GridSearchCV

import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

class Clustering(Report):
    
    def __init__(self, df, name = None):
        
        
        super(Clustering, self).__init__(name)
        
        self.X_train = open_csv(df)
        
    
    def runDBSCAN(self, eps = 0.001, min_samples =2, label = 'DBSCAN'):
        
        dbscan = DBSCAN(eps = eps, metric='euclidean', min_samples=min_samples)
        dbscan.fit(self.X_train)
        
        labels = list(dbscan.labels_)
        
        print (f"{len(set(labels))} clusters have been found by DBSCAN")
        self.log.info(f"{len(set(labels))} clusters have been found by DBSCAN")
        
        PCA2D(self.X_train, labels, self.directory, label)
        PCA3D(self.X_train, labels, self.directory, label)
        
        self.X_train['DBSCAN'] = labels
        self.X_train.to_csv(self.directory + "/"+ self.name + '-DBSCAN.csv')
        
        
        plt.clf()
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)
        
        x = [i for i in range(len(self.X_train))]
        plt.step(x, labels, transform = rot + base)
        plt.show()
        plt.close()
    
    def searchVariables(self, grid= {'eps' : [int(x) for x in np.linspace(start = 1, stop = 100, num = 20)],
                                     'min_samples' : [x for x in range(2,10)]}):
        return
        
    
    
def run():
    
    myClustering = Clustering('Encoder-OnTrain', 'CPTTrain-Encoded')
#     myClustering.findOptimalClusters( min_num_clusters = 1, max_num_clusters=10)
    myClustering.runDBSCAN(eps = 0.06, min_samples=4)

if __name__ == "__main__":
    run()        
        
