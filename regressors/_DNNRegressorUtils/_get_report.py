import numpy as np
from ._shap_deep_regression import shap_deep_regression

def get_report(voters, slicer, utils, **params):

	X_train = params.pop("X_train")
	X_cv = params.pop("X_cv")
	X_test = params.pop("X_test")
	Y_train = params.pop("Y_train")
	Y_cv = params.pop("Y_cv")
	Y_test = params.pop("Y_test")
	dates_train = params.pop("dates_train")
	dates_cv = params.pop("dates_cv")
	dates_test = params.pop("dates_test")
	directory = params.pop("directory")
	model_name = params.pop("model_name")
	interpret = params.pop("interpret", False)
	should_report_all_voters = params.pop("should_report_all_voters", False)
	n_top_features = params.get('n_top_features')
	dl = params.get("dl")

	_, X_test_report, _, Y_test_report, \
				_, dates_test_report = dl.load_with_test()


	all_y_train_pred = []
	all_y_cv_pred = []
	all_y_test_pred = []

	for i, voter in enumerate(voters):

		y_pred_train = voter.predict(X_train).reshape(1,-1)[0]
		y_pred_cv = voter.predict(X_cv).reshape(1, -1)[0]
		y_pred_test = voter.predict(X_test).reshape(1,-1)[0]

		all_y_train_pred.append(y_pred_train)
		all_y_cv_pred.append(y_pred_cv)
		all_y_test_pred.append(y_pred_test)

		if should_report_all_voters or len(voters) == 1:

			utils.evaluate_regression(['OnTrain', X_train, Y_train, dates_train],
								['OnCV', X_cv, Y_cv, dates_cv],
								['OnTest', X_test, Y_test, dates_test],
								['OnCVTest', X_test_report, Y_test_report, dates_test_report],
								direc = directory,
								model = voter,
								model_name = model_name + f"-Voter{i}",
								logger = dl.log,
								slicer = slicer,
								should_check_hetero = True,
								should_log_inverse = dl.should_log_inverse)

	all_y_train_pred = np.mean(np.array(all_y_train_pred), axis = 0)
	all_y_cv_pred = np.mean(np.array(all_y_cv_pred), axis = 0)
	all_y_test_pred = np.mean(np.array(all_y_test_pred), axis = 0)

	if len(voters) > 1:
		utils.evaluate_regression(['OnTrain', X_train, Y_train, all_y_train_pred, dates_train],
								['OnCV', X_cv, Y_cv, all_y_cv_pred, dates_cv],
								['OnTest', X_test, Y_test, all_y_test_pred, dates_test],
								['OnCVTest', X_test_report, Y_test_report, dates_test_report],
								direc = directory,
								model = voter,
								model_name = model_name + f"-Ensemble",
								logger = dl.log,
								slicer = slicer,
								should_check_hetero = dl.should_check_hetero,
								should_log_inverse = dl.should_log_inverse)

	if interpret:

		shap_deep_regression(directory, voters[0], X_train, X_test,
							X_train.columns, n_top_features, dl.log, label = 'DNN-OnTest-Voter0')
		# FIIL(self.directory, self.model, mean_squared_error,
		# 	self.X_test, self.Y_test, self.n_top_features,
		# 	10, self.log, "FIIL")
		