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

# This is a personal logging file for myself, created on January 14, 2019

class Report(object):
    def __init__(self, name = None, model_name = None):
        super(Report, self).__init__()
        
        if not name:
            self.name = input("\nPlease enter the project name: ")
        else:
            self.name = name
        
        self.directory = os.path.join(os.path.dirname(__file__), "XReport", self.name, model_name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        logging_address = os.path.join(self.directory, 'Report.log')
        self.log = Logger(logger_name = self.name + '-Logger', address = logging_address , mode='a')
    
    
    def evaluate_regression(self, y_true, y_pred, inds, label, slicer = 0.1):
        
        # Saviong into csv file
        report = pd.DataFrame()
        report['Actual'] = y_true
        report['Predicted'] = y_pred
        report['Error'] = report['Actual'] - report['Predicted']
        report['Ind'] = inds
        report.set_index('Ind', inplace=True)
        report.to_csv(self.directory + "/"+ f'{label}.csv')
        
        report_str = f"{label}, CorCoef= {CorCoef(y_true, y_pred):.2f}, R2= {R2(y_true, y_pred):.2f}, RMSE={mean_squared_error(y_true, y_pred)**0.5:.2f}, MSE={mean_squared_error(y_true, y_pred):.2f}, MAE={mean_absolute_error(y_true, y_pred):.2f}, MAPE={MAPE(y_true, list(y_pred)):.2f}%"
        
        self.log.info(report_str)
        print(report_str)
        
        # Plotting errors
        errs = list(report['Error'])
        x = [i for i in range(len(errs))]
        plt.clf()
        plt.ylabel('Erros')
        plt.title(label+'-Errors')
        plt.scatter(x, errs, s = 1)
        plt.grid(True)
        plt.savefig(self.directory + '/' + label + '-Errors.png')
        plt.close()

        if slicer != 1:
            y_true, y_pred = y_true[-int(slicer*len(y_true)):], y_pred[-int(slicer*len(y_pred)):]
        
        # let's order them
        temp_list = []
        for true, pred in zip(y_true, y_pred):
            temp_list.append([true, pred])
        temp_list = sorted(temp_list , key=lambda x: x[0])
        y_true, y_pred = [], []
        for i, pair in enumerate(temp_list):
            y_true.append(pair[0])
            y_pred.append(pair[1])
            
        # Actual vs Predicted plotting
        plt.clf()
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(label+'-Actual vs. Predicted')
        ac_vs_pre = plt.scatter(y_true, y_pred, s = 1)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=0.75)
        plt.grid(True)
        plt.savefig(self.directory + '/' + label + '-ACvsPRE.png')
        plt.close()
        
        # Actual and Predicted Plotting
        plt.clf()
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(label + "-Actual and Predicted")
        x = [i for i in range(len(y_true))]
        act = plt.plot(x, y_true, label = "actual")
        pred = plt.plot (x, y_pred, label = 'predicted')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.directory + '/' + label + '-ACandPRE.png')
        plt.clf()
        plt.close()

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
    
    def report_feature_importance(self, best_features, num_top_features, label = "Test" ):
    
        if type(best_features) == list:
            # We only have the most important feature names, not sure how much importanct
            self.log.info(f"Feature importance based on {label}\n" + pprint.pformat(best_features))
            
        elif type(best_features) == dict:
            print ("About to conduct feature importance")
            for k in best_features.keys():
                best_features[k] = abs(best_features[k])
            features_ = pd.Series(OrderedDict(sorted(best_features.items(), key=lambda t: t[1], reverse =True)))
            
            self.log.info(f"Feature importance based on {label}\n" + pprint.pformat(features_.nlargest(num_top_features)))
        
            ax = features_.nlargest(num_top_features).plot(kind='bar', title = label)
            fig = ax.get_figure()
            fig.savefig(self.directory + "/"+ f'{label}-FS.png')
            del fig
            plt.close()
        
        elif isinstance(best_features, pd.Series):
            features_ = best_features
            self.log.info(f"Feature importance based on {label}\n" + pprint.pformat(features_.nlargest(num_top_features)))
            ax = features_.nlargest(num_top_features).plot(kind='bar', title = label)
            fig = ax.get_figure()
            fig.savefig(self.directory + "/"+ f'FS-{label}.png')
            del fig
            plt.close()
        
        else:
            raise TypeError("--- Incompatibale type of features")
    
    def shap_deep_regression(self, model, x_train, x_test, cols, num_top_features = 10, label = 'DNN-OnTest'):
    
        explainer = shap.DeepExplainer(model, x_train.values)
        shap_values = explainer.shap_values(x_test.values)
    
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
    
    def lime_classification(self, x_train, cols, classes):
        
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=cols, class_names=classes, discretize_continuous=True)
    
    
class PlotLosses(tf.keras.callbacks.Callback):
    
    def __init__(self, num):
        self.num = num
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.ion()
        plt.clf()
        plt.title(f'Step {self.num}-epoch:{epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        pointer = -100 if self.i > 100 else 0
        
        plt.plot(self.x[pointer:], self.losses[pointer:], label="loss")
        plt.plot(self.x[pointer:], self.val_losses[pointer:], label = 'cv_loss')
        
        plt.legend()
        plt.grid(True, which = 'both')
        plt.draw()
        plt.pause(0.000001)
    
    def closePlot(self):
        plt.close()

class Logger(object):
    
    instance = None

    def __init__(self, logger_name = 'Logger', address = '',
                 level = logging.DEBUG, console_level = logging.ERROR,
                 file_level = logging.DEBUG, mode = 'w'):
        super(Logger, self).__init__()
        if not Logger.instance:
            logging.basicConfig()
            
            Logger.instance = logging.getLogger(logger_name)
            Logger.instance.setLevel(level)
            Logger.instance.propagate = False
    
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            Logger.instance.addHandler(console_handler)
            
            file_handler = logging.FileHandler(address, mode = mode)
            file_handler.setLevel(file_level)
            formatter = logging.Formatter('%(asctime)s-%(levelname)s- %(message)s')
            file_handler.setFormatter(formatter)
            Logger.instance.addHandler(file_handler)
    
    def _correct_message(self, message):
        output = "\n----------------------------------------------------------\n"
        output += message
        output += "\n---------------------------------------------------------\n"
        return output
        
    def debug(self, message):
        Logger.instance.debug(self._correct_message(message))

    def info(self, message):
        Logger.instance.info(self._correct_message(message))

    def warning(self, message):
        Logger.instance.warning(self._correct_message(message))

    def error(self, message):
        Logger.instance.error(self._correct_message(message))

    def critical(self, message):
        Logger.instance.critical(self._correct_message(message))

    def exception(self, message):
        Logger.instance.exception(self._correct_message(message))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print (f'---- {method.__name__} is about to start ----')
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print (f'---- {method.__name__} is done in {te-ts:.2f} seconds ----')
        return result
    return timed
      
def PCA3D(X, labels = [], address = "", label= ""):      
        

        X = X.values
            
        fig = plt.figure()
        ax = Axes3D(fig)
        
        pca = PCA(n_components=3).fit(X)
        A = pca.transform(X)
        
        if len(labels) == 0:
            A = np.array(A)
            print (A)
            ax.scatter(A[:,0],A[:,1],A[:,2])
        else:
        
            for i in range(0, A.shape[0]):
                if labels[i] == 0:
                    c1 = ax.scatter(A[i,0],A[i,1],A[i,2],c='r',marker='+')
                elif labels[i] == 1:
                    c2 = ax.scatter(A[i,0],A[i,1],A[i,2],c='g', marker='o')
                elif labels[i] == -1:
                    c3 = ax.scatter(A[i,0],A[i,1],A[i,2],c='b', marker='*', label = 'Outliers')
                elif labels[i] == 2:
                    c4 = ax.scatter(A[i,0],A[i,1],A[i,2],c='K', marker='8')
                elif labels[i] == 3:
                    c4 = ax.scatter(A[i,0],A[i,1],A[i,2],c='y', marker='h')
                elif labels[i] == 4:
                    c4 = ax.scatter(A[i,0],A[i,1],A[i,2],c='m', marker='x')
                elif labels[i] == 5:
                    c4 = ax.scatter(A[i,0],A[i,1],A[i,2],c='C', marker='8')
            
        plt.grid(True, which='both')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title(f"{label}-3D")
        ax.legend()
        plt.savefig(address + "/"+ f'{label}-3D.png')
        plt.close()
        
def PCA2D(X, labels = [], address = "", label = ""):
    
    X = X.values
    pca = PCA(n_components=2).fit(X)
    A = pca.transform(X)
    
    if len(labels) == 0:
        labels = [0 for i in range(len(X))]
    
    for i in range(0, A.shape[0]):
        if labels[i] == 0:
            c1 = plt.scatter(A[i,0],A[i,1],c='r',marker= '+')
        elif labels[i] == 1:
            c2 = plt.scatter(A[i,0],A[i,1],c='g', marker='o')
        elif labels[i] == -1:
            c3 = plt.scatter(A[i,0],A[i,1],c='b', marker='*', label = 'Outliers')
        elif labels[i] == 2:
            c4 = plt.scatter(A[i,0],A[i,1],c='K', marker='8')
        elif labels[i] == 3:
            c4 = plt.scatter(A[i,0],A[i,1],c='y', marker='h')
        elif labels[i] == 4:
            c4 = plt.scatter(A[i,0],A[i,1],c='m', marker='x')
        elif labels[i] == 5:
            c4 = plt.scatter(A[i,0],A[i,1],c='C', marker="8")
        
    plt.grid(True, which='both')
    plt.title(f"{label}-2D")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig(address + "/"+ f'{label}-2D.png')
    plt.close()

def prepare_data_simple(df, split_size = 0.2, is_cv = False, is_test = True, should_shuffle = True, is_imbalanced = False):
    
    if isinstance(df, str):
            df = open_csv(df)
            print ("File is loaded")
    
    if isinstance(df, pd.DataFrame):
        
        df = df.iloc[:,:]
        
        X = df.iloc[:,:-1]
        Y = df.iloc[:,-1]
        dates = df.index
        
        if is_cv and is_test:
            print ("With cross validation")
            X_train, X_temp, Y_train, Y_temp, dates_train, dates_temp = train_test_split(X, Y, dates, test_size=split_size, shuffle=should_shuffle, stratify = None, random_state = None)
            X_cv, X_test, Y_cv, Y_test, dates_cv, dates_test = train_test_split(X_temp, Y_temp, dates_temp, test_size = 0.5, shuffle = should_shuffle, random_state = None)
            
            ### Over sampling
            if is_imbalanced:
                print ('About to over sample')
                smt = SMOTE()
                
                X_train, Y_train = smt.fit_resample(X_train, Y_train)
                dates_train = ['SMOTE'+str(i) for i in range(len(X_train))]
                
            return X_train, X_cv, X_test, Y_train, Y_cv, Y_test, dates_train, dates_cv, dates_test
            
        elif is_test:
            X_train, X_test, Y_train, Y_test, dates_train, dates_test = train_test_split(X, Y, dates, test_size=split_size, shuffle=should_shuffle, stratify = None, random_state = None)
            
             ### Over sampling
            if is_imbalanced:
                print ("About to over sample")
                smt = SMOTE()
                
                X_train, Y_train = smt.fit_resample(X_train, Y_train)
                dates_train = ['SMOTE'+str(i) for i in range(len(X_train))]
                
            return X_train, X_test, Y_train, Y_test, dates_train, dates_test
        
        elif not is_cv and not is_test:
            return X, Y, dates
        
        else:
            raise TypeError("Wrong types for the is_cv and is_test")
            
    elif isinstance(df, list) and len(df) == 2:
        X_train = df[0].iloc[:,:-1]
        Y_train = df[0].iloc[:,-1]
        dates_train = df[0].index
        
        X_test = df[1].iloc[:,:-1]
        Y_test = df[1].iloc[:,-1]
        dates_test = df[1].index
        return X_train, X_test, Y_train, Y_test, dates_train, dates_test

    elif isinstance(df, list) and len(df) == 3:
        X_train = df[0].iloc[:,:-1]
        Y_train = df[0].iloc[:,-1]
        dates_train = df[0].index
        
        X_cv = df[1].iloc[:,:-1]
        Y_cv = df[1].iloc[:,-1]
        dates_cv = df[1].index
        
        X_test = df[2].iloc[:,:-1]
        Y_test = df[2].iloc[:,-1]
        dates_test = df[2].index
    
        return X_train, X_cv, X_test, Y_train, Y_cv, Y_test, dates_train, dates_cv, dates_test
    
    else:
        raise ValueError ("--- Incompatible df format")

 def sensitivity_vahid(self, label = ""):
        
    x = self.X_test.copy()
    y = self.Y_test.copy()
    cols = x.columns
    
    model_error = mean_squared_error(self.model.predict(x), y)
    print (f'Model Error:{model_error:.2f}')
    
    feature_importances_ = {}
    for col in cols:
        x_temp = x.copy()
        temp_err = []
        for _ in range(100):
            np.random.shuffle(x_temp[col].values)
            err = model_error - mean_squared_error(self.model.predict(x_temp), y)
            temp_err.append(err)
        feature_importances_[col] = abs(round(np.mean(temp_err), 4)) if np.mean(temp_err)<0 else 0
    
    self.report_feature_importance(feature_importances_, self.num_top_features, label = f'{label}-Vahid-FIIL' )

def open_csv(file_name):
    dir = os.path.dirname(__file__) + "/_data_bin/"+ file_name + ".csv"
    df = pd.read_csv(dir, index_col = 0)
    return df.copy()

def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except ZeroDivisionError:
        return 0

def R2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]**2

def CorCoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]


if __name__ == "__main__":
    
    print(open_csv("SoilNoOut"))
    
    myLogger = Logger(address = "log.log")
    myLogger.info('Hi')
