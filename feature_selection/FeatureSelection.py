import seaborn as sns
import pymrmr
from pyHSICLasso import HSICLasso
from collections import OrderedDict
from ReliefF import ReliefF

from sklearn_relief import RReliefF as rf
import sklearn_relief as relief

import skfeature

REGRESSION, CLASSIFICATION = 'regression', 'classification'

import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_regression, f_classif, mutual_info_classif, mutual_info_regression

from sklearn.model_selection import train_test_split

        
class FeatureSelection(Report):
    
    def __init__(self, dfs = pd.DataFrame(),
                 name = None,
                 split_size = 0.05,
                 should_shuffle = True,
                 num_top_features = 10,
                 type = CLASSIFICATION):
        
        super(FeatureSelection, self).__init__(name, 'FeatureSelection')
        
        self.num_top_features = num_top_features
        
        self.type = type
        
        #splitting data into X and Y
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.dates_train, self.dates_test = prepare_data_simple(dfs, split_size, is_cv = False, should_shuffle = should_shuffle)
        self.cols = self.X_train.columns
        
        self.data = self.X_train.copy()
        self.data['Y'] = self.Y_train
        
    
    @timeit
    def selectKBest(self):
        
        def get_features(cols, scores):
            feats = {}
            for col, scr in zip(cols, scores):
                feats[col] = scr
            return feats
        
        if self.type == REGRESSION:
            print ("Feature selection with f_regression metrics is about to be conducted")
            FS = SelectKBest(f_regression, k = self.num_top_features).fit(self.X_train, self.Y_train)
            self.report_feature_importance(get_features(self.cols, FS.scores_), self.num_top_features, label = "f_regression" )
            
            print ("Feature selection with mutual_info_regression metrics is about to be conducted")
            FS = SelectKBest(mutual_info_regression, k = self.num_top_features).fit(self.X_train, self.Y_train)
            self.report_feature_importance(get_features(self.cols, FS.scores_), self.num_top_features, label = "mutual_info_regression" )
            
        elif self.type== CLASSIFICATION:
            print ("Feature selection with f_classif metrics is about to be conducted")
            FS = SelectKBest(f_classif, k = self.num_top_features).fit(self.X_train, self.Y_train)
            self.report_feature_importance(get_features(self.cols, FS.scores_), self.num_top_features, label = "f_classif" )
            
            print ("Feature selection with mutual_classification metrics is about to be conducted")
            FS = SelectKBest(mutua_info_classif, k = self.num_top_features).fit(self.X_train, self.Y_train)
            self.report_feature_importance(get_features(self.cols, FS.scores_), self.num_top_features, label = "mutual_info_classification" )
    
    @timeit
    def correlation(self):
        
        features_ = {}
        for col in self.X_train.columns:
            features_[col] = round(np.corrcoef(self.Y_train, self.X_train[col])[0,1], 4)
        
        self.report_feature_importance(features_, self.num_top_features, label = "Correlation" )
        
        
        sns.set(font_scale=20)
        corrmat = self.data.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(200,200))
        #plot heat map
        grid = sns.heatmap(self.data[top_corr_features].corr(), annot=False, cmap="RdYlGn", annot_kws={"size": 500}, cbar_kws={"shrink": 1}, xticklabels=True, yticklabels=True, square=True)
        grid.tick_params(labelsize = 300)
        grid.set_xlabel("Features" ,fontsize=300, labelpad= 600)
        grid.set_ylabel("Features" ,fontsize=300, labelpad= 600)
        
        fig = grid.get_figure()
        fig.savefig(self.directory + "/CorrHeatMap-"+ self.name + '.png')
        del fig
        plt.close()
        
        mpl.style.use('classic')
            
    @timeit
    def mRMR(self):
        
        df_ = self.data.copy()
        cols = list(df_.columns)[:-1] + ['class']
        df_.columns = cols
        
        if self.type == CLASSIFICATION:
            features_ = pymrmr.mRMR(df_, 'MID', self.num_top_features)
            self.report_feature_importance(features_, self.num_top_features, label = "mMRM - MID" )
            
            features_ = pymrmr.mRMR(df_, 'MIQ', self.num_top_features)
            self.report_feature_importance(features_, self.num_top_features, label = "mMRM - MIQ" )
        else:
            print ("mRMR is designed to be used in for classification, not regression ")
            
    @timeit
    def HSICLasso(self):
        
        df_ = self.data.copy()
        cols = list(df_.columns)[:-1] + ['class']
        df_.columns = cols
    
        hsic_lasso = HSICLasso()
        hsic_lasso.input(self.X_train.values, self.Y_train.values)
        
        if self.type == CLASSIFICATION:
            hsic_lasso.classification(self.num_top_features)
        elif self.type == REGRESSION:
            hsic_lasso.regression(self.num_top_features)
        
        feats = [df_.columns[int(val)-1] for val in hsic_lasso.get_features()]
        
        for feat, imp in zip(feats, hsic_lasso.get_index_score()):
            features_[feat] = imp
        self.report_feature_importance(features_, self.num_top_features, label = "HSICLasso" )
    
    @timeit
    def RReliefF(self):
        
        r = relief.RReliefF(n_features=self.num_top_features)
        transformed_matrix = r.fit_transform(self.X_train.values, self.Y_train.values)
        
        feats = self.X_train.columns 
        features_ =  dict(zip(feats, r.w_))
        self.report_feature_importance(features_, self.num_top_features, label = "RReliefF" )
        


@timeit
def run():
    
    file = 'Diabetes1'
    
    fs = FeatureSelection(file,
                name = file,
                 split_size = 0.05,
                 should_shuffle = True,
                 num_top_features = 20,
                 type = REGRESSION)
    fs.correlation()
    fs.selectKBest()
#     fs.mRMR()
#     fs.HSICLasso()
    fs.RReliefF()
    

if __name__ == "__main__":
    run()

        