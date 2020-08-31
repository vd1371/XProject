import pandas as pd
import matplotlib.pyplot as plt
import os
import easygui

import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.utils import np_utils
from keras.models import model_from_json

from keras.regularizers import l1, l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from keras.layers import Dropout

        
class Classifier(Report):
    
    """
        in case of any problem
        https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        
        Assumptions:
            1. It is assumed that Y is placed in the last column of the data set
    """

    def __init__(self, dataset = pd.DataFrame(),
                 name = None,
                 classes = [-1,0,1],
                 should_shuffle = True,
                 split_size = 0.4):
        
        super(Classifier, self).__init__(name, 'DNN')
    
            
        self.dataset = shuffle(dataset) if should_shuffle else dataset
        #finding the number of dataset columns
        self.number_of_cols = len(self.dataset.columns)
        
        #splitting data into X and Y
        X = self.dataset.iloc[:,:self.number_of_cols-1]
        Y = self.dataset.iloc[:,self.number_of_cols-1]
        
        dates = self.dataset.index
        
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)

        # Splitting the original Y
        self.Y_original_train, Y_original_temp = train_test_split(Y, test_size = split_size, shuffle=False)
        self.Y_original_cv, self.Y_original_test = train_test_split(Y_original_temp, test_size = 0.5, shuffle=False)
         
        #splitting data into train, cross_validation, test
        self.X_train, X_temp, self.Y_train, Y_temp, self.train_dates, temp_dates = train_test_split(X, dummy_y, dates, test_size=split_size, shuffle=False)
        X_cv, X_test, self.Y_cv, self.Y_test, self.cv_dates, self.test_dates = train_test_split(X_temp, Y_temp, temp_dates, test_size = 0.5, shuffle=False)
        self.X_cv, self.X_test = X_cv.values, X_test.values

        self.classes = [str(i) for i in classes]
        self.number_of_classes = len(classes)
    
    def setLayers(self,layers):
        self.layers=layers
    
    def setInputActivationFunction(self, activation_function):
        self.input_activation_func = activation_function
    
    def setHiddenActivationFunction(self, hidden_activation_func):
        self.hidden_activation_func = hidden_activation_func
    
    def setFinalActivationFunction(self, final_activation_func):
        self.final_activation_func = final_activation_func
    
    def setLossFunction(self, loss_func):
        self.loss_func = loss_func
    
    def setEpochs(self, epochs):
        self.epochs = epochs
        
    def setMinDelta(self, min_delta):
        self.min_delta = min_delta
        
    def setPatience(self, patience):
        self.patience = patience
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        
    def shouldEarlyStop(self, val):
        self.should_early_stop = val
    
    def shouldPlot(self, val):
        self.should_plot_live_error = val       
        
    def setReg(self, reg_param, type='l1'):
        self.l = l1 if type == 'l1' else l2
        self.reg_param = reg_param 
    
    def runTrainingCurve(self, steps = 10, should_save_fig = True):
        
        
        print ("\nTrainig curve is about to be produced....\nIt will take a while, plese be patient...\n")
        self.log.info("\nTrainig curve is about to be produced....\nIt will take a while, plese be patient...\n")
        
        l = len(self.X_train)
        
        cv_errors, train_errors = [], []
        
        for i in range(1,steps+1):
        
            model = 0
        
            # Split rows for training curves
            indexer = int((i/steps)*l)
            X_train = self.X_train[:indexer]
            Y_train = self.Y_train[:indexer]
             
            #creating the structure of the neural network
            model = Sequential()
            model.add(Dense(self.layers[0], input_dim = self.number_of_cols-1, activation = self.input_activation_funciton))
            for ind in range(1,len(self.layers)):
                model.add(Dense(self.layers[ind], activation = self.hidden_activation_func))
            model.add(Dense(self.number_of_classes, activation=self.final_activation_func))
             
            # Compile model
            model.compile(loss=self.loss_func, optimizer='Adam')
             
            # Creating Early Stopping function and other callbacks
            call_back_list = []
            early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
            plot_losses = PlotLosses(i)
        
            if self.should_early_stop:
                call_back_list.append(early_stopping)
            if self.should_plot_live_error:
                call_back_list.append(plot_losses) 
             
            # Fit the model
            X_train = X_train.values
            model.fit(X_train, Y_train, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses.closePlot()
            
            # Evaluate the model
            train_scores = model.evaluate(X_train, Y_train, verbose=2)
            cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=2)
            
            print ("Step", i, "training loss:" , train_scores, "cross validation loss:", cv_scores)
            self.log.info("Step %d, training loss:%0.5f, cross validation loss:%0.5f" %(i, train_scores, cv_scores))
             
            # Add errors to list
            train_errors.append(train_scores)
            cv_errors.append(cv_scores)
             
            print ("---Step %d is done--\n---------------------\n" %(i))
        
        
        # Serialize model to JSON
        save_address = self.directory + "/" + self.name 
        model_json = model.to_json()
        with open(save_address + ".json" , "w") as json_file:
            json_file.write(model_json)
        # Serialize weights to HDF5
        model.save_weights(save_address + ".h5")
        print ("---------------------\nModel is Saved")

        
        # Creating X values for plot
        x_axis = [i for i in range(1,steps+1)]
        
        # Plot
        plt.clf()
        plt.xlabel('Step')
        plt.ylabel('Error - MSE')
        plt.title(self.name+'Training Curve')
        plt.plot(x_axis, cv_errors, label = 'CV-err')
        plt.plot(x_axis, train_errors, label = 'Train-err')
        
        plt.legend()
        plt.grid(True)
        
        if should_save_fig:
            plt.savefig(self.directory + '/TC-' + self.name + '.png')
        plt.show()
        
    def runRegularizationParameterAnalysis(self, first_guess = 0.001, final_value = 3, increment = 2, should_save_fig = True):
        
        # Creating empty list for errors
        cv_errors, train_errors = [], []
        
        X_train, Y_train = self.X_train.values, self.Y_train
        
        
        reg = first_guess
        while reg < final_value:
            
            #creating the structure of the neural network
            model = Sequential()
             
            model.add(Dense(self.layers[0], input_dim = self.number_of_cols-1, activation = self.input_activation_funciton, kernel_regularizer=self.l(reg)))
            for ind in range(1,len(self.layers)):
                model.add(Dense(self.layers[ind], activation = self.hidden_activation_func, kernel_regularizer=self.l(reg)))
            model.add(Dense(self.number_of_classes, activation = self.final_activation_func, kernel_regularizer=self.l(reg)))
             
            # Compile model
            model.compile(loss=self.loss_func, optimizer='Adam')
            
            # Creating Early Stopping function and other callbacks
            call_back_list = []
            early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
            plot_losses = PlotLosses(reg)
        
            if self.should_early_stop:
                call_back_list.append(early_stopping)
            if self.should_plot_live_error:
                call_back_list.append(plot_losses)
            
            # Fit the model
            model.fit(X_train, Y_train, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size,
                      shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses = PlotLosses(reg)
             
            # evaluate the model
            train_scores = model.evaluate(X_train, Y_train, verbose=0)
            cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=0)
             
            cv_errors.append(cv_scores)
            train_errors.append(train_scores)
            
            print (f"err_training: {train_scores:0.4f}, cv_err: {cv_scores:0.4f}")
            self.log.info("\nerr_training:%0.4f, cv_err:%0.4f\n" %(train_scores, cv_scores))
            
            print ("---Fitting with %s as regularization parameter is done---" %str(reg))
            self.log.info("\n---Fitting with %s as regularization parameter is done---\n" %str(reg))
            
            reg = reg*increment
             
        
        
        x_axis = [(i+1) for i in range(len(cv_errors))]
        
        plt.clf()
        plt.xlabel('Regularization Paremeter Step')
        plt.ylabel('Error - MSE')
        plt.title(self.name+'Regularization Analysis')
        plt.plot(x_axis, cv_errors, label = 'CV-err')
        plt.plot(x_axis, train_errors, label = 'Train-err')
        
        plt.legend()
        plt.grid(True)
        
        if should_save_fig:
            plt.savefig(self.directory + '/RA-' + self.name + '.png')
        plt.show()
    
    def runDropOutAnalysis(self, first_guess = 0.1, final_value = 0.9, increment = 0.1, should_save_fig = True):
        
        # Creating empty list for errors
        cv_errors, train_errors = [], []
        
        X_train, Y_train = self.X_train.values, self.Y_train
        
        
        drop = first_guess
        while drop < final_value:
            
            #creating the structure of the neural network
            model = Sequential()
             
            model.add(Dense(self.layers[0], input_dim = self.number_of_cols-1, activation = self.input_activation_funciton, kernel_dropularizer=dropularizers.l2(drop)))
            model.add(Dropout(drop))
            for ind in range(1,len(self.layers)):
                model.add(Dense(self.layers[ind], activation = self.hidden_activation_func))
                model.add(Dropout(drop))
            model.add(Dense(self.number_of_classes, activation = self.final_activation_func))
             
            # Compile model
            model.compile(loss=self.loss_func, optimizer='Adam')
            
            # Creating Early Stopping function and other callbacks
            call_back_list = []
            early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
            plot_losses = PlotLosses(drop)
        
            if self.should_early_stop:
                call_back_list.append(early_stopping)
            if self.should_plot_live_error:
                call_back_list.append(plot_losses)
            
            # Fit the model
            model.fit(X_train, Y_train, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size,
                      shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses = PlotLosses(drop)
             
            # evaluate the model
            train_scores = model.evaluate(X_train, Y_train, verbose=0)
            cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=0)
             
            cv_errors.append(cv_scores)
            train_errors.append(train_scores)
            
            print ("err_training:", train_scores, "cv_err:", cv_scores)
            self.log.info("\nerr_training:%0.4f \ncv_err:%0.4f\n" %(train_scores, cv_scores))
            
            drop = drop + increment
             
            print ("---Fitting with %s as drop parameter is done---" %str(drop))
            self.log.info("\n---Fitting with %s as drop parameter is done---\n" %str(drop))
        
        
        x_axis = [(i+1) for i in range(len(cv_errors))]
        
        plt.clf()
        plt.xlabel('drop Paremeter Step')
        plt.ylabel('Error - MSE')
        plt.title(self.name+'drop Analysis')
        plt.plot(x_axis, cv_errors, label = 'CV-err')
        plt.plot(x_axis, train_errors, label = 'Train-err')
        
        plt.legend()
        plt.grid(True)
        
        if should_save_fig:
            plt.savefig(self.directory + '/RA-' + self.name + '.png')
        plt.show()
    
    def loadModel(self):
        
        # load json and create model
        address = self.directory + "/" + self.name 
        json_file = open(address + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights(address+ ".h5")
        self.model.compile(loss=self.loss_func, optimizer='Adam')
    
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
    
    def fitModel(self, drop=0.5):
        
        
        X_train, Y_train = self.X_train.values, self.Y_train
        
        #creating the structure of the neural network
        model = Sequential()
        model.add(Dense(self.layers[0],
                        input_dim = self.number_of_cols-1,
                        activation = self.input_activation_func,
                        kernel_regularizer=self.l(self.reg_param)))
        model.add(Dropout(drop))
        
        for ind in range(1,len(self.layers)):
            model.add(Dense(self.layers[ind],
                            activation = self.hidden_activation_func,
                            kernel_regularizer=self.l(self.reg_param)))
            model.add(Dropout(drop))
            
        model.add(Dense(self.number_of_classes, activation = self.final_activation_func, kernel_regularizer=self.l(self.reg_param)))
         
        # Compile model
        model.compile(loss=self.loss_func, optimizer='Adam')
        
        model.summary(print_fn=self.log.info)
        
        # Creating Early Stopping function and other callbacks
        call_back_list = []
        early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
        plot_losses = PlotLosses(0)
    
        if self.should_early_stop:
            call_back_list.append(early_stopping)
        if self.should_plot_live_error:
            call_back_list.append(plot_losses) 
         
        # Fit the model
        X_train = self.X_train.values
        model.fit(X_train, Y_train, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2, callbacks=call_back_list)
        plot_losses.closePlot()
        
        model.summary()
        
        # Evaluate the model
        train_scores = model.evaluate(X_train, Y_train, verbose=2)
        cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=2)
        test_scores = model.evaluate(self.X_test, self.Y_test, verbose=2)
        
        print ("err_training:", train_scores, "cv_err:", cv_scores, "test_err", test_scores)
        self.log.info("\nerr_training: %0.4f \ncv_err: %0.4f \ntest_err: %0.4f\n" %(train_scores, cv_scores, test_scores))
        
        # Serialize model to JSON
        save_address = self.directory + "/" + self.name 
        model_json = model.to_json()
        with open(save_address + ".json" , "w") as json_file:
            json_file.write(model_json)
        # Serialize weights to HDF5
        model.save_weights(save_address + ".h5")
        print ("---------------------\nModel is Saved")
        
    def predict_set(self, X, threshold = 0.3, neutral_class = '0'):
        
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
    
    def get_report(self, threshold = 0.5, neutral_class = '0'):
        

        
        self.evaluate_classification(self.Y_original_train, self.predict_set(self.X_train), self.train_dates, 'OnTrain')
        self.evaluate_classification(self.Y_original_test, self.predict_set(self.X_test), self.test_dates, 'OnTest')
        
    def test_out_data(self):
        
        ### This is temporary, you can use it for other files
        df  =pd.read_csv('D:\\Academics\\AIProject\\1-DigitRecongnition\\Data\\test.csv', index_col = 0)
        
        pred = self.predict_set(df)
        
        df['Label'] = pred
        
        df.to_csv('D:\\Academics\\AIProject\\1-DigitRecongnition\\Data\\test_predicted.csv')
        
    
        
@timeit       
def run():
    data = pd.read_csv("D:\\Academics\Xproject\_data_bin\Digit_train.csv", index_col = 0)
    print (data)
    
    myClassifier = Classifier(data.iloc[:,:], name = "Digit_train", classes = [0,1,2,3,4,5,6,7,8,9], should_shuffle=True, split_size=0.2)
    
    myClassifier.setLayers([1200, 800, 400, 400, 400, 800, 1200])
    myClassifier.setLossFunction('categorical_crossentropy')
    myClassifier.setEpochs(500)
    
    myClassifier.setInputActivationFunction('sigmoid')
    myClassifier.setHiddenActivationFunction('relu')
    myClassifier.setFinalActivationFunction('softmax')
    
    myClassifier.shouldPlot(False)
    myClassifier.shouldEarlyStop(True)
    
    myClassifier.setBatchSize(2048)
    myClassifier.setPatience(50)
    myClassifier.setMinDelta(0.001)
    
    myClassifier.setReg(0.00000, 'l2')
    
    
#     myClassifier.runTrainingCurve(steps=10)
#     myClassifier.runRegularizationParameterAnalysis(first_guess = 0.0000001, final_value = 0.000001, increment = 2, should_save_fig=True)
#     myClassifier.runDropOutAnalysis(first_guess=0.1, final_value=0.5, increment=0.1, should_save_fig=True)
    
#     myClassifier.fitModel(drop=0.5) 
    myClassifier.loadModel()
#     myClassifier.test_out_data()
#     myClassifier.thresholOptimization(neutral_class=0, objective_classes=[1], increment = 0.01)
    myClassifier.get_report(threshold = 0.3, neutral_class= '0')

if __name__ == "__main__":
    run()

        