
def sort_ys_based_on_y_true(y_true, y_pred):

	temp_list = list(zip(y_true, y_pred))
	temp_list = sorted(temp_list , key=lambda x: x[0])
	y_true, y_pred = list(zip(*temp_list))

	return y_true, y_pred

