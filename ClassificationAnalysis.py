import matplotlib
matplotlib.use('Agg')

from utils.DataLoader import DataLoader
from utils.AwesomeLogger import Logger

def logistic_regression(name, data_loader):
	from classifiers.LogisticRegression import Logit

	logit = Logit(name , data_loader)
	logit.fit()

def tree_classifier(name, data_loader):
	from classifiers.TreesClassifier import Trees
	
	tr = Trees(name, data_loader)
	tr.set_params(n_estimators = 50,
					max_depth = None,
					min_samples_split=2,
					min_samples_leaf=1,
					max_features='auto',
					should_cross_val=False,
					n_jobs = -1)
	
	tr.fit_decision_tree()
	tr.fit_random_forest()
	# tr.fit_extra_trees()
	# tr.fit_balanced_random_forest()

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

def boostigs(name, data_loader):
	from classifiers.BoostingClassifier import Boosting

	bst = Boosting(name, data_loader)
	bst.set_params(	max_depth = 12,
					n_estimators = 2000,
					learning_rate = 0.1,
					gamma = 0,
					min_child_weight = 1,
					n_jobs = -1,
					verbose = 1,
					objective = 'multi:softmax',
					# objective = 'binary:logistic',
					reg_alpha = 0,
					reg_lambda = 0.0000001,
					n_iter = 2000)

	bst.xgb_run()
	bst.catboost_run(warm_up = False)

def svm_classifier(name, data_loader):
	from classifiers.SVC import SVMC

	sv = SVMC(name, data_loader)
	sv.fit()

def random_classifier(name, data_loader):
	from classifiers.RandomClassifier import RandomClassifier

	rm = RandomClassifier(name, data_loader)
	rm.fit()

def dnn_classifier(name, data_loader):

	from classifiers.DNNClassifier import DNNC
	myClassifier = DNNC(name, 'DNN', data_loader)
	myClassifier.set_classes([1, 2, 3])

	myClassifier.set_layers([200, 200])
	myClassifier.set_loss_function("binary_crossentropy")
	myClassifier.set_epochs(500)

	myClassifier.set_input_activation_function('tanh')
	myClassifier.set_hidden_activation_function('relu')
	myClassifier.set_final_activation_function('sigmoid')
	
	myClassifier.set_optimizer('Adam')
	
	myClassifier.should_plot_live(False)
	myClassifier.should_early_stop(False)
	myClassifier.should_checkpoint(False)
	
	myClassifier.set_batch_size(2048)
	myClassifier.set_patience(500)
	myClassifier.set_min_delta(2)
	myClassifier.set_reg(0.0000001, 'l2')
	
#     myClassifier.runLearningCurve(steps=10)
#     myClassifier.runRegularizationParameterAnalysis(first_guess = 0.000001, final_value = 0.002, increment=3)
	myClassifier.fit_model(drop=0, warm_up = False)
	myClassifier.get_report()

if __name__ == "__main__":

	import os
	import numpy as np
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	file_name = 'Impact4'

	dl = DataLoader(df = file_name,
					split_size = 0.2,
					should_shuffle = True,
					is_imbalanced = True,
					random_state = 65,
					k = 5,
					n_top_features = 20,
					n_samples = None,
					should_log_inverse = False,
					modelling_type = 'c')

	logistic_regression(file_name, dl)
	tree_classifier(file_name, dl)
	# svm_classifier(file_name, dl)
	boostigs(file_name, dl)
	# knn_regressor(file_name, dl)
	# dnn_classifier(file_name, dl)
	# random_classifier(file_name, dl)