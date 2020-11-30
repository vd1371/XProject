import logging, sys, time, pprint, datetime, itertools, os, sys

import pandas as pd
import numpy as np
import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import joblib

from collections import OrderedDict

import shap
from imblearn.over_sampling import SMOTE
import lime, lime.lime_tabular

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score

from builtins import isinstance

import lime
import lime.lime_tabular

def evaluate_classification(self, y_true, y_pred, inds, label, extra_df = None):

    self.log.info(f"----------Classification Report for {label}------------\n" + str(classification_report(y_true, y_pred))+"\n")
    self.log.info(f"----------Confusion Matrix for {label}------------\n" + str(confusion_matrix(y_true, y_pred))+"\n")
    self.log.info(f'----------Accurcay for {label}------------\n'+str(round(accuracy_score(y_true, y_pred),4)))
    
    print (classification_report(y_true, y_pred))
    print (f'Accuracy score for {label}', round(accuracy_score(y_true, y_pred),4))
    print ("------------------------------------------------")
    
    
    report = pd.DataFrame()
    report['Actual'] = y_true
    report['Predicted'] = y_pred
    report['Ind'] = inds
    report.set_index('Ind', inplace=True)
    report.to_csv(self.directory + "/" + f'{label}.csv')
    
    if isinstance(extra_df, pd.DataFrame):
        df = pd.concat([report, extra_df], axis = 1, join = 'inner')
        df.to_csv(self.directory + "/" + f'{label}-ExtraInformation.csv')

    
def shap_deep_classification(self, model, x_train, x_test, cols, num_top_features = 10, label = 'DNN-OnTest'):

    explainer = shap.DeepExplainer(model, x_train.values)
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values[0], features = x_test, feature_names=cols, show=False)
    plt.savefig(self.directory + f"/ShapValues-{label}.png")
    plt.close()
    
    shap_values = pd.DataFrame(shap_values[0], columns = list(x_train.columns)).abs().mean(axis = 0)
    self.log.info(f"SHAP Values {label}\n" + pprint.pformat(shap_values.nlargest(num_top_features)))
    
    ax = shap_values.nlargest(num_top_features).plot(kind='bar', title = label)
    fig = ax.get_figure()
    
    fig.savefig(self.directory + "/"+ f'ShapValuesBar-{label}.png')
    del fig
    plt.close()
