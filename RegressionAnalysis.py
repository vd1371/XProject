import matplotlib
matplotlib.use('Agg')

from utils.DataLoader import DataLoader 

def tree_regressor(data_loader):
	from regressors.TreesRegressors import Trees
	
	tr = Trees(data_loader)
	tr.set_params(n_estimators = 100, max_depth = None, min_samples_split=2,
				 min_samples_leaf=1, max_features='auto', should_cross_val=True)

	tr.fit_random_forest()
	tr.fit_decision_tree()
	tr.fit_extra_trees()

	# tr.tune_rees(RandomForestClassifier,
	# 			'RF',
	# 			grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)],
	# 			'max_features': ['auto', 'sqrt'],
	# 			'max_depth': [int(x) for x in np.linspace(3, 20, num = 16)] + [None],
	# 			'min_samples_split': [val for val in np.linspace(start = 0.1, stop = 0.9, num = 9)],
	# 			'min_samples_leaf': [val for val in np.linspace(start = 0.1, stop = 0.5, num = 5)],
	# 			'bootstrap': [True, False]},
	# 			should_random_search = True,
	# 			n_iter = 1)

def svm_regressor(data_loader):
	from regressors.SVR import SVMR
	
	mySVM = SVMR(data_loader)
	mySVM.set_params(C = 10000, kernel='rbf', epsilon=0.2, gamma='auto')
	mySVM.fit_model()
	# mySVM.regularization_analysis(start = 1, end = 100000, step = 3, kernel = 'rbf')
	# mySVM.epsilon_analysis(C=50)

def linear_regressor(data_loader):
	from regressors.LinearRegressors import Linear

	lin = Linear(data_loader)
	lin.set_params(alpha = 0.002,
					fit_intercept = True,
					should_cross_val = False)
	# lin.fit_ols()
	lin.fit_linear()
	# lin.fit_ridge()
	# lin.fit_lasso()
	# lin.analyze(model_name = 'Lasso', start = 0.000001, end=100, step=2)
	# lin.fit('lasso', alpha = 0.002, should_analyze= False, start = 0.000001, end = 2, step = 2)
	# lin.fit('ridge', alpha = 0.002, should_analyze= False, start = 0.000001, end = 2, step = 2)

def knn_regressor(data_loader):
	from regressors.KNNRegressor import KNNR

	myKNNR = KNNR(data_loader)
	myKNNR.fit_model(n = 5)
	# myKNNR.neighbour_analysis(start=3, end=20, step=2)

def dnn_regressor(data_loader):

	from regressors.DNNRegressor import DNNR
	myRegressor = DNNR('DNN', data_loader)
	myRegressor.set_hyperparameters(n_voters = 1,
									layers = [40, 40, 40],
									input_activation_func = 'tanh',
									hidden_activation_func = 'relu',
									final_activation_func = 'linear',
									loss_func = 'mape',
									# loss_func = 'mean_absolute_percentage_error',
									# loss_func = 'mean_squared_logarithmic_error',
									epochs = 1000,
									min_delta = 2,
									patience = 50,
									batch_size = 32,
									should_early_stop = False,
									should_plot_live_error = False,
									should_checkpoint = False,
									regul_type = 'l2',
									reg_param = 0.000001,
									optimizer = 'Adam',
									dropout = 0,)

	myRegressor.fit_model(warm_up = False)
	myRegressor.get_report(slicer = 1, interpret = True)

if __name__ == '__main__':

	import os
	import numpy as np
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	project_name = "Hacka"
	file_name = 'Hacka2-Scaled-Ran'
	# file_name = 'Hacka2-Scaled-Ran-WithoutFinancials'
	dl = DataLoader(project_name = project_name,
					df = (file_name, None),
					split_size = 0.3,
					should_shuffle=True,
					is_imbalanced=False,
					random_state = 165,
					k = 10,
					n_top_features = 100,
					n_samples = None,
					should_log_inverse = False,
					should_check_hetero = True,
					modelling_type = 'r')

	# linear_regressor(dl)
	# tree_regressor(dl)
	svm_regressor(dl)
	# knn_regressor(dl)
	# dnn_regressor(dl)
