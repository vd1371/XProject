import numpy as np
import joblib

from utils.BaseModel import BaseModel
from utils.SpecialPlotters import train_cv_analyzer_plotter
from utils.AwesomeTimeIt import timeit
from utils.RegressionReport import evaluate_regression
from utils.FeatureImportanceReport import report_feature_importance
from utils.PlotLosses import PlotLosses

import keras
from keras.models import Sequential, load_model
import keras.losses
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras.models import model_from_json

import matplotlib.pyplot as plt

        
class DNNR(BaseModel):

    def __init__(self, name, dl):
        
        super().__init__(name, 'DNN')
        
        self.n_top_features = dl.n_top_features
        self.k = dl.k
        
        self.X_train, self.X_cv, self.X_test, \
            self.Y_train, self.Y_cv, self.Y_test, \
            self.dates_train, self.dates_cv, self.dates_test = dl.load_with_csv()
        
        self.X, self.Y, _ = dl.load_all()

        self.input_dim = len(self.X_train.columns)
        
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
    
    def set_reg(self, reg_param, type='l1'):
        self.l = l1 if type == 'l1' else l2
        self.reg_param = reg_param
    
    def set_optimizer(self,val):
        self.optimizer = val

    def _construct_model(self, reg = None):

        if reg is None:
            reg = self.reg_param

        model = Sequential()
        model.add(Dense(self.layers[0], input_dim = self.input_dim, activation = self.input_activation_func, kernel_regularizer=self.l(reg)))
        for ind in range(1,len(self.layers)):
            model.add(Dense(self.layers[ind], activation = self.hidden_activation_func, kernel_regularizer=self.l(reg)))
        model.add(Dense(1, activation = self.final_activation_func))
         
        # Compile model
        model.compile(loss=self.loss_func, optimizer=self.optimizer)
         
        # Creating Early Stopping function and other callbacks
        call_back_list = []
        early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
        plot_losses = PlotLosses()
    
        if self.should_early_stop:
            call_back_list.append(early_stopping)
        if self.should_plot_live_error:
            call_back_list.append(plot_losses)

        return model, call_back_list
    
    @timeit
    def run_learning_curve(self, steps = 10):
        
        l = len(self.X_train)
        
        cv_errors, train_errors = [], []
        
        for i in range(1,steps+1):
        
            model = 0
        
            # Split rows for Learning curves
            indexer = int((i/steps)*l)
            X_train = self.X_train[:indexer]
            Y_train = self.Y_train[:indexer]
             
            #creating the structure of the neural network
            model, call_back_list = self._construct_model(reg = 0)
            
            # Fit the model
            model.fit(X_train.values, Y_train.values, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses.closePlot()
            
            print (f"Step {i} of run_learning_curve is done")
             
            # Add errors to list
            train_errors.append(model.evaluate(X_train, Y_train, verbose=2))
            cv_errors.append(model.evaluate(self.X_cv, self.Y_cv, verbose=2))
             
        train_cv_analyzer_plotter(train_errors, cv_errors, self.directory, 'TrainingCurve', xticks = None)

        
    def run_regularization_parameter_analysis(self, first_guess = 0.001, final_value = 3, increment = 2, should_save_fig = True):
        
        # Creating empty list for errors
        cv_errors, train_errors, xticks = [], [], []
        
        reg = first_guess
        while reg < final_value:
            
            xticks.append(f"{reg:.2E}")
            #creating the structure of the neural network
            model, call_back_list = self._construct_model(reg = reg)
                
            # Fit the model
            model.fit(self.X_train.value, self.Y_train.values, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses = PlotLosses(reg)
            
            train_errors.append(model.evaluate(X_train, Y_train, verbose=0))
            cv_errors.append(model.evaluate(self.X_cv, self.Y_cv, verbose=0))
            
            print (f"---Fitting with {reg:.2E} as regularization parameter is done---")
            
            reg = reg*increment
             
        train_cv_analyzer_plotter(train_errors, cv_errors, self.directory, 'TrainingCurve', xticks = None)
        
        
        
    def fit_model(self, drop = 0.1):
        
        # #creating the structure of the neural network
        model, call_back_list = self._construct_model()
         
        # Creating Early Stopping function and other callbacks
        call_back_list = []
        early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
        plot_losses = PlotLosses()
    
        if self.should_early_stop:
            call_back_list.append(early_stopping)
        if self.should_plot_live_error:
            call_back_list.append(plot_losses)
        
        # a summary of the model
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.log.info(short_model_summary)
        
        # Fit the model
        model.fit(self.X_train.values, self.Y_train.values, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, verbose = 2, shuffle=True, callbacks=call_back_list)
        plot_losses.closePlot()
        
        # Evaluate the model
        train_scores = model.evaluate(self.X_train.values, self.Y_train.values, verbose=2)
        cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=2)
        test_scores = model.evaluate(self.X_test, self.Y_test, verbose=2)
            
        print ()
        print (f'Trian_err: {train_scores:.2f}, Cv_err: {cv_scores:.2f}, Test_err: {test_scores:.2f}')
        self.log.info(f'Trian_err: {train_scores:.2f}, Cv_err: {cv_scores:.2f}, Test_err: {test_scores:2f}')

        self.save_model(model)
        
    def load_model(self):
        
        # load json and create model
        address = self.directory + "/" +  self.name
        self.model = load_model(address+ ".h5")
        self.model.compile(loss=self.loss_func, optimizer=self.optimizer)

    def save_model(self, model):
        save_address = self.directory + "/" + self.name 
        model.save(save_address + ".h5")

        
    def get_report(self, slicer = 0.5):
        
        y_pred_train = self.model.predict(self.X_train).reshape(1,-1)[0]
        y_pred_cv = self.model.predict(self.X_cv).reshape(1, -1)[0]
        y_pred_test = self.model.predict(self.X_test).reshape(1,-1)[0]
        
        evaluate_regression(self.directory, self.Y_train, y_pred_train, self.dates_train, 'DNN-OnTrain', self.log, slicer=slicer)
        evaluate_regression(self.directory, self.Y_cv, y_pred_cv, self.dates_cv, 'DNN-OnCV', self.log, slicer=slicer)
        evaluate_regression(self.directory, self.Y_test, y_pred_test, self.dates_test, 'DNN-OnTest', self.log, slicer=slicer)

        
        
@timeit
def run():
    
    myRegressor = DNNR(file_name, dl)
    
    myRegressor.set_layers([400])
    myRegressor.set_loss_function('MSE')
    myRegressor.set_epochs(200)
    
    myRegressor.set_input_activation_function('tanh')
    myRegressor.set_hidden_activation_function('relu')
    myRegressor.set_final_activation_function('sigmoid')
    
    myRegressor.set_optimizer('Adam')
    
    myRegressor.should_plot(False)
    myRegressor.should_early_stop(False)
    
    myRegressor.set_batch_size(4096)
    myRegressor.set_patience(100)
    myRegressor.set_min_delta(2)
    myRegressor.set_reg(0.000003, 'l2')
    
#     myRegressor.run_learning_curve(steps=10)
#     myRegressor.run_regularization_parameter_analysis(first_guess = 0.000001, final_value = 0.002, increment=3)
    myRegressor.fit_model(drop=0)
    myRegressor.load_model()
    myRegressor.get_report(slicer = 1)
    myRegressor.sensitivity_vahid()

if __name__ == "__main__":
    run()

        