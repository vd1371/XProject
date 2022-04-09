import numpy as np

def get_x_y_true_pred(ls, **params):
	
	model = params.get('model')
	should_log_inverse = params.get('should_log_inverse', False)

	if len(ls) == 4:
		label, x, y_true, inds = ls
		y_pred = model.predict(x)
	elif len(ls) == 5:
		label, x, y_true, y_pred, inds = ls

	#For the output of the DNN regression
	if len(np.shape(y_pred)) > 1:
		y_pred = y_pred.reshape(1,-1)[0]

	if should_log_inverse:
		y_true = np.exp(y_true)
		y_pred = np.exp(y_pred)

	return label, x, y_true, y_pred, inds