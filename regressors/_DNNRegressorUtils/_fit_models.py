import numpy as np
from ._save_history_of_model import save_history_of_model
from ._mix_and_split import mix_and_split
from ._close_live_plot import close_live_plot

def fit_models(callbacks, **params):

	voters = params.get('voters')

	dl = params.get('dl')
	epochs = params.get("epochs")
	batch_size = params.get("batch_size")
	logger = params.get("log")
	should_report_all_voters = params.pop("should_report_all_voters", False)

	x_train = params.get("X_train").values
	y_train = params.get("Y_train").values
	x_cv = params.get("X_cv")
	y_cv = params.get("Y_cv")
	x_test = params.get("X_test")
	y_test = params.get("Y_test")

	train_scores = []
	cv_scores = []
	test_scores = []


	for i, (voter, callback_list) in enumerate(zip(voters, callbacks)):

		print (f"\nAbout to fit the voter {i+1}/{len(voters)}...\n")
		if len(voters) > 0:
			x_train, x_cv, y_train, y_cv = \
					mix_and_split(x_train, x_cv, y_train, y_cv)

		hist = voter.fit(x_train, y_train,
						validation_data = (x_cv, y_cv),
						epochs = epochs,
						batch_size = batch_size,
						verbose = 2,
						shuffle=True,
						callbacks=callback_list)

		if should_report_all_voters or len(voters) == 1:
			save_history_of_model(hist, i, **params)

		close_live_plot(callback_list)

		# Evaluate the model
		train_scores.append(voter.evaluate(x_train, y_train, verbose=0))
		cv_scores.append(voter.evaluate(x_cv, y_cv, verbose=0))
		test_scores.append(voter.evaluate(x_test, y_test, verbose=0))

	train_scores = np.mean(train_scores)
	cv_scores = np.mean(cv_scores)
	test_scores = np.mean(test_scores)
			
	print (f'Trian_err: {train_scores}, Cv_err: {cv_scores}, Test_err: {test_scores}')
	logger.info(f'Trian_err: {train_scores}, Cv_err: {cv_scores}, Test_err: {test_scores}')

	return voters