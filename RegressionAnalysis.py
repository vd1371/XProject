from utils.DataLoader import DataLoader 

def tree_regressor(name, data_loader):
	from regressors.TreesRegressors import Trees
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import ExtraTreesRegressor
	
	tr = Trees(name, data_loader)
	tr.set_params(n_estimators = 100, max_depth = 5, min_samples_split=2,
				 min_samples_leaf=1, max_features='auto', should_cross_val=False)

	# tr.set_trees({'RF':RandomForestRegressor, 'ETC':ExtraTreesRegressor, 'DT':DecisionTreeRegressor})
	tr.set_trees({'RF':RandomForestRegressor})
	# tr.set_trees({'DT':DecisionTreeRegressor})

	# tr.tune_rees(grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)],
	#                      'max_features': ['auto', 'sqrt'],
	#                      'max_depth': [int(x) for x in np.linspace(3, 20, num = 16)] + [None],
	#                      'min_samples_split': [val for val in np.linspace(start = 0.1, stop = 0.9, num = 9)],
	#                      'min_samples_leaf': [val for val in np.linspace(start = 0.1, stop = 0.5, num = 5)],
	#                      'bootstrap': [True, False]},
	#                      should_random_search = True,
	#                      n_iter = 1)
	tr.fit_all()

def svm_regressor(name, data_loader):
	from regressors.SVR import SVMR
	
	mySVM = SVMR(name, data_loader)
	
	mySVM.fit_model(C=1000000, kernel='rbf', epsilon=0.2, gamma='auto')
	# mySVM.regularization_analysis(start = 1, end = 100000, step = 3, kernel = 'rbf')
	# mySVM.epsilon_analysis(C=50)

def linear_regressor(name, data_loader):
	from regressors.LinearRegressors import Linear

	lin = Linear(name, data_loader)
	lin.fit_ols()
	lin.fit('linear', should_analyze= True, start = 0.000001, end = 2, step = 2)
	# lin.fit('lasso', alpha = 0.002, should_analyze= False, start = 0.000001, end = 2, step = 2)
	# lin.fit('ridge', alpha = 0.002, should_analyze= False, start = 0.000001, end = 2, step = 2)

def knn_regressor(name, data_loader):
	from regressors.KNNRegressor import KNNR

	myKNNR = KNNR(name, data_loader)
	myKNNR.fit_model(n = 5)
	# myKNNR.neighbour_analysis(start=3, end=20, step=2)

def dnn_regressor(name, data_loader):

	from regressors.DNNRegressor import DNNR
	myRegressor = DNNR(name, data_loader)
	
	myRegressor.set_layers([40, 40, 40])
	# myRegressor.set_loss_function('mean_absolute_percentage_error')
	# myRegressor.set_loss_function('mean_squared_logarithmic_error')
	myRegressor.set_loss_function('mape')
	myRegressor.set_epochs(1000)
	
	myRegressor.set_input_activation_function('tanh')
	myRegressor.set_hidden_activation_function('relu')
	myRegressor.set_final_activation_function('linear')
	
	myRegressor.set_optimizer('Adam')
	
	myRegressor.should_plot_live(False)
	myRegressor.should_early_stop(False)
	myRegressor.should_checkpoint(False)
	
	myRegressor.set_batch_size(32)
	myRegressor.set_patience(500)
	myRegressor.set_min_delta(2)
	myRegressor.set_reg(0.000001, 'l2')
	
#     myRegressor.runLearningCurve(steps=10)
#     myRegressor.runRegularizationParameterAnalysis(first_guess = 0.000001, final_value = 0.002, increment=3)
	myRegressor.fit_model(drop=0, warm_up = False)
	myRegressor.get_report(slicer = 1)

if __name__ == '__main__':

	import os
	import numpy as np
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	
	file_name = 'Hacka2-Scaled'
	dl = DataLoader(file_name, split_size = 0.2,
								should_shuffle=True,
								is_imbalanced=False,
								random_state = 165,
								k = 5,
								n_top_features = 40,
								n_samples = None,
								should_log_inverse = False,
								modelling_type = 'r')

	# linear_regressor(file_name, dl)
	# tree_regressor(file_name, dl)
	# svm_regressor(file_name, dl)
	# knn_regressor(file_name, dl)
	dnn_regressor(file_name, dl)

