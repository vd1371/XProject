import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import gc

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import model_from_json
from keras.utils import np_utils
from keras.models import model_from_json


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


class NeuralNet(object):
    
    """
        in case of any problem
        https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        
        Assumptions:
            1. It is assumed that Y is placed in the last column of the data set
    """
    
    def __init__(self):
        super(NeuralNet, self).__init__()

    def setParams(self,
                x_train, x_test,
                y_train, y_test,
                ml_type = 'Regressor',
                 layers = [10, 10],
                 epochs = 2000,
                 activation_func = 'tanh',
                 final_activation_func = 'linear',
                 loss_func = 'mean_squared_error',
                 classes = [1,0],
                 should_early_stop = True,
                 min_delta = 0.0005,
                 patience = 50,
                 batch_size = 50,
                 split_size = 0.4,
                 drop = 1,
                 reg_param = 0.000):
        
        super(NeuralNet, self).__init__()
        self.ml_type = ml_type
        
        self.dim = len(x_train.columns)
        self.drop = drop
        self.reg_param = reg_param
        
        
        # Status: not checked
        
        if self.ml_type == 'Regressor':
            
            #splitting data into train, cross_validation, test
            self.x_train, self.x_test = x_train.values, x_test.values
            self.y_train, self.y_test = y_train.values, y_test.values
        
        elif self.ml_type == 'Classifier':
            
            self.number_of_classes = len(classes)
            
            #TODO : The classifier section needs a whole new implementation. Modify it in the future
            
            # encode class values as integers
#             encoder = LabelEncoder()
#             encoder.fit(Y)
#             encoded_Y = encoder.transform(Y)
#             # convert integers to dummy variables (i.e. one hot encoded)
#             dummy_y = np_utils.to_categorical(encoded_Y)
#     
#             # Splitting the original Y
#             self.Y_original_train, Y_original_temp = train_test_split(Y, test_size = split_size, shuffle=False)
#             self.Y_original_cv, self.Y_original_test = train_test_split(Y_original_temp, test_size = 0.5, shuffle=False)
#              
#             #splitting data into train, cross_validation, test
#             self.X_train_total, X_temp, self.Y_train_total, Y_temp = train_test_split(X, dummy_y, test_size=split_size, shuffle=False)
#             X_cv, X_test, self.Y_cv, self.Y_test = train_test_split(X_temp, Y_temp,test_size = 0.5, shuffle=False)
#             self.X_cv, self.X_test = X_cv.values, X_test.values
        
        else:
            print ("Incorrect ml_type")
        
        self.min_delta = min_delta
        self.patience = patience
        self.batch_size = batch_size
        
        self.layers = layers
        self.activation_funciton = activation_func
        self.final_activation_func = final_activation_func
        self.loss_func = loss_func
        self.epochs = epochs
        self.should_early_stop = should_early_stop
        
    
    def fit_regressor(self):
        
        # STATUS: not checked
        
        #creating the structure of the neural network
        model = Sequential()
        model.add(Dense(self.layers[0],
                        input_dim = self.dim,
                        activation = self.activation_funciton,
                        kernel_regularizer=regularizers.l1(self.reg_param)))
#         model.add(Dropout(self.drop))
        for ind in range(1,len(self.layers)):
            model.add(Dense(self.layers[ind],
                            activation = 'relu',
                            kernel_regularizer=regularizers.l1(self.reg_param)))
            model.add(Dropout(self.drop))
        model.add(Dense(1, activation = self.final_activation_func, kernel_regularizer=regularizers.l1(self.reg_param)))
         
        # Compile model
        model.compile(loss=self.loss_func, optimizer='Adam')
         
        # Creating Early Stopping function and other callbacks
        call_back_list = []
        early_stopping = EarlyStopping(monitor='val_loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto')
        plot_losses = PlotLosses('Test')
        
        if self.should_early_stop:
            call_back_list.append(early_stopping)
        if False:
            call_back_list.append(plot_losses)
        
         
        # Fit the model
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size, shuffle=False, verbose=0, callbacks=call_back_list)
        
        # Evaluate the model
        train_scores = model.evaluate(self.x_train, self.y_train, verbose=0)
        test_scores = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        del model
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        gc.collect()
        
        return train_scores, test_scores

    
    def fitClassifier(self):
        
        # STATUS: need a whole new implementation
        
        X_train_total, Y_train_total = self.X_train_total.values, self.Y_train_total
        
        #creating the structure of the neural network
        model = Sequential()
        model.add(Dense(self.layers[0],
                        input_dim = self.number_of_cols-1,
                        activation = self.activation_funciton,
                        kernel_regularizer=regularizers.l2(self.reg_param)))
        model.add(Dropout(self.drop))
        
        for ind in range(1,len(self.layers)):
            model.add(Dense(self.layers[ind],
                            activation = 'relu',
                            kernel_regularizer=regularizers.l2(self.reg_param)))
#             model.add(Dropout(self.drop))
            
        model.add(Dense(self.number_of_classes, activation = self.final_activation_func, kernel_regularizer=regularizers.l2(self.reg_param)))
         
        # Compile model
        model.compile(loss=self.loss_func, optimizer='Adam')
         
        # Creating Early Stopping function and other callbacks
        call_back_list = []
        early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto')
    
        if self.should_early_stop:
            call_back_list.append(early_stopping)
         
        # Fit the model
        X_train = self.X_train_total.values
        model.fit(X_train_total, Y_train_total, epochs=self.epochs, batch_size=self.batch_size, shuffle=False, verbose=0, callbacks=call_back_list)
        
        # Evaluate the model
        train_scores = model.evaluate(X_train_total, Y_train_total, verbose=0)
        cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=0)
        test_scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        
        return np.mean([train_scores, cv_scores, test_scores])

class PlotLosses(keras.callbacks.Callback):
    
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
        
        pointer = -200 if self.i > 200 else 0
        
        plt.plot(self.x[pointer:], self.losses[pointer:], label="loss")
        plt.plot(self.x[pointer:], self.val_losses[pointer:], label = 'cv_loss')
        
        plt.legend()
        plt.grid(True, which = 'both')
        plt.draw()
        plt.pause(0.00001)
    
    def closePlot(self):
        plt.close()
        
def run():
    data = pd.read_csv("FirstTryTest.csv", index_col = 0)
    myNeural = predictor(data, ml_type = 'Regressor')
    myNeural.fitModel(reg_param=0.000, drop=0.5)

if __name__ == "__main__":
    print ("This should not e run")

        