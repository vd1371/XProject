import os
import pprint
import numpy as np
import pandas as pd
import joblib
import time

from utils.BaseModel import BaseModel
from utils.AwesomeTimeIt import timeit
from utils.FeatureImportanceReport import report_feature_importance
from utils.ClassificationReport import evaluate_classification
from utils.FeatureImportanceReport import report_feature_importance
from utils.PlotLosses import PlotLosses

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import model_from_json

from keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        
class DNNC(BaseModel):

    def __init__(self, name, model_name, dl):
        
        super().__init__(name, model_name, dl)
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k

        self.X_train, self.X_cv, self.X_test, \
            self.Y_train, self.Y_cv, self.Y_test, \
                self.Y_original_train, self.Y_original_cv, self.Y_original_test, \
                    self.dates_train, self.dates_cv, self.dates_test = dl.load_for_classification()

        self.input_dim = len(self.X_train.columns)

    def set_classes(self, classes):
        self.classes = [str(i) for i in classes]
        self.number_of_classes = len(classes)

    def set_layers(self,layers):
        self.layers=layers
    
    def set_input_activation_function(self, activation_function):
        self.input_activation_func = activation_function
    
    def set_hidden_activation_function(self, hidden_activation_func):
        self.hidden_activation_func = hidden_activation_func
    
    def set_final_activation_function(self, final_activation_func):
        self.final_activation_func = final_activation_func
    
    def set_loss_function(self, loss_func):
        self.loss_func = loss_func
    
    def set_epochs(self, epochs):
        self.epochs = epochs
        
    def set_min_delta(self, min_delta):
        self.min_delta = min_delta
        
    def set_patience(self, patience):
        self.patience = patience
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def should_early_stop(self, val):
        self.should_early_stop = val
    
    def should_plot_live(self, val):
        self.should_plot_live_error = val

    def should_checkpoint(self, val):
        self.should_checkpoint = val     
        
    def set_reg(self, reg_param, regul_type='l1'):
        self.regul_type = regul_type
        self.l = l1 if regul_type == 'l1' else l2
        self.reg_param = reg_param
    
    def set_optimizer(self,val):
        self.optimizer = val

    def _get_call_backs(self):
        # Creating Early Stopping function and other callbacks
        call_back_list = []
        early_stopping = EarlyStopping(monitor='loss',
                                        min_delta = self.min_delta,
                                        patience=self.patience,
                                        verbose=1,
                                        mode='auto') 
        plot_losses = PlotLosses()
    
        if self.should_early_stop:
            call_back_list.append(early_stopping)
        if self.should_plot_live_error:
            call_back_list.append(plot_losses)
        if self.should_checkpoint:
            checkpoint = ModelCheckpoint(os.path.join(self.directory,
                                                    f'{self.name}-BestModel.h5'),
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='auto')
            call_back_list.append(checkpoint)

        return call_back_list

    def log_hyperparameters(self):
        self.log.info(pprint.pformat({'layers': self.layers,
                                    'input_activation_func': self.input_activation_func,
                                    'hidden_activation_func': self.hidden_activation_func,
                                    'final_activation_func': self.final_activation_func,
                                    'loss_func': self.loss_func,
                                    'epochs': self.epochs,
                                    'min_delta': self.min_delta,
                                    'patience': self.patience,
                                    'batch_size': self.batch_size,
                                    'should_early_stop': self.should_early_stop,
                                    'regularization_type': self.regul_type,
                                    'reg_param': self.reg_param,
                                    'random_state': self.dl.random_state}))

    def _construct_model(self, reg = None):

        self.log_hyperparameters()

        if reg is None:
            reg = self.reg_param

        model = Sequential()
        model.add(Dense(self.layers[0],
                        input_dim = self.input_dim,
                        activation = self.input_activation_func,
                        kernel_regularizer=self.l(reg)))
        for ind in range(1,len(self.layers)):
            model.add(Dense(self.layers[ind],
                            activation = self.hidden_activation_func,
                            kernel_regularizer=self.l(reg)))
        model.add(Dense(self.number_of_classes, activation = self.final_activation_func))
         
        # Compile model
        model.compile(loss=self.loss_func,
                        optimizer=self.optimizer,
                        metrics = ['CategoricalAccuracy'])

        return model
    
    @timeit
    def run_learning_curve(self, steps = 10):
        
        l = len(self.X_train)
        
        cv_errors, train_errors = [], []
        
        for i in range(1,steps+1):
        
            model = 0
        
            # Split rows for training curves
            indexer = int((i/steps)*l)
            X_train = self.X_train[:indexer]
            Y_train = self.Y_train[:indexer]
             
            #creating the structure of the neural network
            model, call_back_list = self._construct_model(reg = 0), self._get_call_bakcs()
             
            # Fit the model
            model.fit(X_train.values, Y_train.values,
                        validation_data=(self.X_cv, self.Y_cv),
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        verbose=2,
                        callbacks=call_back_list)
            plot_losses.closePlot()
            
            # Evaluate the model
            train_scores = model.evaluate(X_train, Y_train, verbose=2)
            cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=2)
             
            # Add errors to list
            train_errors.append(train_scores)
            cv_errors.append(cv_scores)
             
            print (f"---Step {d} is done---")
        
        train_cv_analyzer_plotter(train_errors, cv_errors, self.directory, 'TrainingCurve', xticks = None)
        
    def run_regularization_parameter_analysis(self, first_guess = 0.001,
                                                    final_value = 3,
                                                    increment = 2):
        
        # Creating empty list for errors
        cv_errors, train_errors, xticks = [], [], []
        
        reg = first_guess
        while reg < final_value:

            xticks.append(f"{reg:.2E}")
            #creating the structure of the neural network
            model = self._construct_model(reg = reg)
            call_back_list = self._get_call_bakcs()
            
            # Fit the model
            model.fit(self.X_train.value, self.Y_train.values,
                        validation_data=(self.X_cv, self.Y_cv),
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses = PlotLosses(reg)
             
            # evaluate the model
            train_errors.append(model.evaluate(X_train, Y_train, verbose=0))
            cv_errors.append(model.evaluate(self.X_cv, self.Y_cv, verbose=0))

            print (f"---Fitting with {reg:.2E} as regularization parameter is done---")
            reg = reg*increment
    
        train_cv_analyzer_plotter(train_errors, cv_errors, self.directory, 'TrainingCurve', xticks = None)
    
    def fit_model(self, drop=0, warm_up = False):
        
        constructed = False
        if warm_up:
            try:
                self.load_model()
                constructed = True
                self.log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
            except OSError:
                print ("The model is not trained before. No saved models found")
        
        if not constructed:
            # Creating the structure of the neural network
            self.model = self._construct_model()
            
            # A summary of the model
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            self.log.info(short_model_summary)

        call_back_list = self._get_call_backs()

        start = time.time()
        # Fit the model
        hist = self.model.fit(self.X_train, self.Y_train,
                            validation_data=(self.X_cv, self.Y_cv),
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            verbose = 2, shuffle=True, callbacks=call_back_list)

        # Logging call_back history
        hist_df = pd.DataFrame.from_dict(hist.history)
        hist_df.to_csv(f"{self.directory}/{self.loss_func}-hist.csv")
        print (f"********* {time.time()-start:.4f} ***********")

        # Closing the plot losses
        try:
            for call_back in call_back_list:
                call_back.closePlot()
        except:
            pass
    
        # Evaluate the model
        train_scores = self.model.evaluate(self.X_train, self.Y_train, verbose=2)
        cv_scores = self.model.evaluate(self.X_cv, self.Y_cv, verbose=2)
        test_scores = self.model.evaluate(self.X_test, self.Y_test, verbose=2)
        
        print ()
        print (f'Trian_err: {train_scores}, Cv_err: {cv_scores}, Test_err: {test_scores}')
        self.log.info(f'Trian_err: {train_scores}, Cv_err: {cv_scores}, Test_err: {test_scores}')

        self.save_model()
    
    def load_model(self):
        
        # load json and create model
        model_type = 'BestModel' if self.should_checkpoint else 'SavedModel'
        self.model = load_model(self.directory + "/" +  f"{self.name}-{model_type}.h5")

    def save_model(self):
        save_address = self.directory + "/" + self.name 
        self.model.save(save_address + "-SavedModel.h5", save_format = 'h5')
    
    def predict_set(self, X, threshold = 0.5, neutral_class = '3'):
        
        # The model should be loaded prior to running this section
        Y_predicted_weighted = self.model.predict(X)
        Y_predicted = []
            
        for pred in Y_predicted_weighted:
                num_of_categorized = 0
                predicted_class = 0
                for cat_pred in pred:
                    if cat_pred >= threshold:
                        num_of_categorized += 1
                        predicted_class = self.classes[list(pred).index(cat_pred)]
                
                if num_of_categorized == 1:
                    Y_predicted.append(predicted_class)
                else:
                    Y_predicted.append(neutral_class)
        
        predicted = list(map(int,Y_predicted[:]))
        
        
        return predicted
    
    def get_report(self):

        self.load_model()

        y_train_pred = self.predict_set(self.X_train)
        y_test_pred = self.predict_set(self.X_test)

        evaluate_classification(['OnTrain', self.X_train, self.Y_original_train, self.dates_train, y_train_pred],
                                ['OnTest', self.X_test, self.Y_original_test, self.dates_test, y_test_pred],
                                direc = self.directory,
                                model = self.model,
                                model_name = self.model_name,
                                logger = self.log,
                                slicer = 1)

    def thresholOptimization(self, neutral_class = '0', objective_classes = [-1,1], increment = 0.02, should_save_fig = True):
        
        # Model should be loaded prior to threshold optimizatio
        
        # Finding the index of objective classes
        objective_classes_indices = [self.classes.index(str(i)) for i in objective_classes ]
        
        # Predict Ys
        Y_predicted_categorical = self.model.predict(self.X_test)
        
        # First define a value to start and Create threshold list
        thresh = 1.0/self.number_of_classes
        threshold_list = []
        while thresh < 1:
            threshold_list.append(thresh)
            thresh += increment
        
        f1_total_list = []
        
        for thresh  in threshold_list:
            # Find the class for each prediction
            Y_predicted = []
            
            print ("Analyzing f1_score if threshold equals to %s" %str(thresh))
            
            for pred in Y_predicted_categorical:
                
                num_of_categorized = 0
                predicted_class = 0
                for cat_pred in pred:
                    if cat_pred >= thresh:
                        num_of_categorized += 1
                        predicted_class = self.classes[list(pred).index(cat_pred)]
                
                if num_of_categorized == 1:
                    Y_predicted.append(int(predicted_class))
                else:
                    Y_predicted.append(int(neutral_class))
    
                
            f1_report = f1_score(Y_predicted, self.Y_original_test, average=None)
            
            f1_param_summation = 0
            for ind in objective_classes_indices:
                f1_param_summation += f1_report[ind]
            f1_total_list.append(f1_param_summation)
        
        
        plt.clf()
        plt.xlabel('Threshold')
        plt.ylabel('f1_score')
        plt.title(self.name+ 'f1_score optimiazation')
        plt.plot(threshold_list, f1_total_list, label = 'f1_score')
        
        plt.legend()
        plt.grid(True)
        
        if should_save_fig:
            plt.savefig(self.directory + '/Threshold-' + self.name + '.png')
        plt.show()