import pandas as pd
import numpy as np

import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
class OutlierDetector(Report):
    
    """
        This modules tries to find outliers in a dataset of continous variables
        Based on multivariate normal distribution
    """

    def __init__(self, df, name = None, percentage = 0.05, thresh = 0.01):
        
        super(OutlierDetector, self).__init__(name, 'Preprocessing')
        
        if isinstance(df, pd.DataFrame):
            self.data = df
        elif isinstance(df, str):
            self.data = open_csv(df)
        else:
            raise TypeError("--- df type is wrong")
        
        self.percentage = percentage
        self.threshold = thresh
        
#         self.sites = self.data.iloc[:,:106]
#         self.data = self.data.iloc[:,-7:]
        
        
        self.log.info('-------------- Outliers are about to be found %s '%self.name)

    def setPercentage(self, per):
        self.percentage = per

    def findOutliers(self, distribution = 'normal'):
        
        if distribution == 'log-normal':
            self.data = self.data.apply(np.log)
        
        # Finding covariance matrix and mean array
        corr_mat = self.data.corr()
        cov_mat = self.data.cov()
        
        mean_arr = [self.data[col_name].astype('float64').mean() for col_name in self.data.columns]
        
        self.log.info("Correlation matrix\n"  +str(corr_mat))
        self.log.info("Mean array:" + str(mean_arr))
        
        # Creating multivariate probability distribution function
        var = multivariate_normal(mean=mean_arr, cov=cov_mat)
        self.data['pdf'] = self.data.apply(lambda row : var.pdf(row), axis = 1)
        
        # Sorting based on pdf
        self.data.sort_values(by=['pdf'], axis = 0, inplace = True, ascending= True)
        
        # Slicing the dataset
        self.data = self.data.iloc[int(self.percentage*len(self.data)):,:].copy()
        
        # Shuffle the dataset and reseting index
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        # Dropping the pdf column
        self.data.drop('pdf', axis = 1, inplace=True)
        
        if distribution == 'log-normal':
            self.data = self.data.apply(np.exp)
        
        # Saving the original dataset
#         self.data = pd.concat([self.sites, self.data], axis = 1, join = 'inner')
        
        self.data.to_csv(self.directory + "/" + self.name + "-NoOutOriginal.csv")
        
#         self.data = self.data.iloc[:,-7:]
        
        print (f'The data for {self.name} is saved')
        
    def standardize(self, method = 'Standard', which_cols = "all", methods_list= []):
        
        pointer = 0 if which_cols == 'all' else 1
        
        if method == 'Standard':
            methods_list = ['S' for _ in range(len(self.data.columns)-pointer)]
        
        elif method == 'Normal':
            methods_list = ['N' for _ in range(len(self.data.columns)-pointer)]
            
        
        cols = self.data.columns[:] if pointer == 0 else self.data.columns[:-1]
        
        # Scaling the Data
        
        for i , col in enumerate(cols):
            
            if methods_list[i] == 'S':
                scaler = StandardScaler()
            elif methods_list[i] == 'N':
                scaler = MinMaxScaler()
                
            self.data[col] = scaler.fit_transform(np.reshape(self.data[col].values, (-1,1)))
                
#         self.data = pd.concat([self.sites, self.data], axis = 1, join = 'inner')
        # Saving the dataset
        self.data.to_csv(self.directory + "/" + self.name + "-NoOutScaled.csv")
        print (f'The data for {self.name} is saved')
        
        
        
def run():
    myDetector = OutlierDetector('DNNPaper','DNNPaper', percentage = 0.05)
    myDetector.findOutliers(distribution='log')
    
    # "Standard" or "Normal"
    # Which call 'all' or else

    myDetector.standardize(method = 'Standard', which_cols = "Not-all")
    

if __name__ == "__main__":
    run()

        