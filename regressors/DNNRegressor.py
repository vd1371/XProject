import os
import numpy as np
import pandas as pd
import utils

from ._DNNRegressorUtils import *
		
class DNNR(utils.BaseModel):

	def __init__(self, model_name, dl):
		
		super().__init__(model_name, dl)
		
		self.n_top_features = dl.n_top_features
		self.k = dl.k
		
		self.X_train, self.X_cv, self.X_test, \
			self.Y_train, self.Y_cv, self.Y_test, \
			self.dates_train, self.dates_cv, self.dates_test = dl.load_with_csv()
		
		self.X, self.Y, _ = dl.load_all()

		self.input_dim = len(self.X_train.columns)

	def set_hyperparameters(self, **params):

		self.n_voters = params.pop("n_voters", 1)
		self.layers = params.pop("layers", [10])
		self.input_activation_func = params.pop("input_activation_func", 'tanh')
		self.hidden_activation_func = params.pop("hidden_activation_func", 'relu')
		self.final_activation_func = params.pop("final_activation_func", "linear")
		self.loss_func = params.pop("loss_func", 'mse')
		self.epochs = params.pop("epochs", 100)
		self.min_delta = params.pop("min_delta", 0.01)
		self.patience = params.pop("patience", 50)
		self.batch_size = params.pop("batch_size", 256)
		self.should_early_stop = params.pop("should_early_stop", False)
		self.should_plot_live_error = params.pop("should_plot_live_error", True)
		self.should_checkpoint = params.pop("should_checkpoint", False)
		self.regul_type = params.pop("regul_type", 'l1')
		self.reg_param = params.pop("reg_param", 0.000001)
		self.optimizer = params.pop("optimizer", 'Adam')
		self.dropout = params.pop("dropout", 0)

		log_hyperparameters(**self.__dict__)
	
	# @utils.timeit
	# def run_learning_curve(self, steps = 10):

	# 	raise ValueError
		
	# 	l = len(self.X_train)
		
	# 	cv_errors, train_errors = [], []
		
	# 	for i in range(1,steps+1):
		
	# 		model = 0
		
	# 		# Split rows for Learning curves
	# 		indexer = int((i/steps)*l)
	# 		X_train = self.X_train[:indexer]
	# 		Y_train = self.Y_train[:indexer]
			 
	# 		#creating the structure of the neural network
	# 		model, call_back_list = self._construct_model(reg = 0), _get_callbacks(**self.__dict__)
			
	# 		# Fit the model
	# 		model.fit(X_train.values, Y_train.values,
	# 					validation_data=(self.X_cv, self.Y_cv),
	# 					epochs=self.epochs,
	# 					batch_size=self.batch_size,
	# 					shuffle=True,
	# 					verbose=2,
	# 					callbacks=call_back_list)
	# 		plot_losses.closePlot()
			
	# 		print (f"Step {i} of run_learning_curve is done")
			 
	# 		# Add errors to list
	# 		train_errors.append(model.evaluate(X_train, Y_train, verbose=2))
	# 		cv_errors.append(model.evaluate(self.X_cv, self.Y_cv, verbose=2))
			 
	# 	utils.train_cv_analyzer_plotter(train_errors, cv_errors, self.directory, 'TrainingCurve', xticks = None)

		
	# def run_regularization_parameter_analysis(self, first_guess = 0.001,
	# 												final_value = 3,
	# 												increment = 2):

	# 	raise ValueError
		
	# 	# Creating empty list for errors
	# 	cv_errors, train_errors, xticks = [], [], []
		
	# 	reg = first_guess
	# 	while reg < final_value:
			
	# 		xticks.append(f"{reg:.2E}")
	# 		#creating the structure of the neural network
	# 		model = self._construct_model(reg = reg)
	# 		call_back_list = _get_callbacks(**self.__dict__)
				
	# 		# Fit the model
	# 		model.fit(self.X_train.value, self.Y_train.values,
	# 					validation_data=(self.X_cv, self.Y_cv),
	# 					epochs=self.epochs,
	# 					batch_size=self.batch_size,
	# 					shuffle=True, verbose=2, callbacks=call_back_list)
	# 		plot_losses = utils.PlotLosses(reg)
			
	# 		train_errors.append(model.evaluate(X_train, Y_train, verbose=0))
	# 		cv_errors.append(model.evaluate(self.X_cv, self.Y_cv, verbose=0))
			
	# 		print (f"---Fitting with {reg:.2E} as regularization parameter is done---")
			
	# 		reg = reg*increment
			 
	# 	utils.train_cv_analyzer_plotter(train_errors, cv_errors, self.directory, 'TrainingCurve', xticks = None)
	
	@utils.timeit
	def fit_model(self, warm_up):

		models = TheModel(**self.__dict__)
		models.construct(warm_up)
		models.fit()
		models.save()

		
	def get_report(self, slicer, interpret = False):

		models = TheModel(**self.__dict__)
		models.load()

		get_report(models.voters,
					slicer,
					utils,
					interpret = interpret,
					**self.__dict__)
		
@utils.timeit
def run():
	
	myRegressor = DNNR(file_name, dl)
	
	myRegressor.set_layers([400])
	myRegressor.set_loss_function('MSE')
	myRegressor.set_epochs(200)
	
	myRegressor.set_input_activation_function('tanh')
	myRegressor.set_hidden_activation_function('relu')
	myRegressor.set_final_activation_function('linear')
	
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

		