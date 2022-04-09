import numpy as np
import matplotlib.pyplot as plt
from ._get_direc_for_label import get_direc_for_label

def plot_hetero(x, y_true, y_pred, label, **params):

	direc = params.get('direc')
	direc = get_direc_for_label(direc, label)
	model_name = params.get('model_name')
	should_check_hetero = params.get('should_check_hetero', True)
	
	# Plotting the error percentage vs. feature and the output variable
	if should_check_hetero:
		try:

			if np.all(y_true):
				error_vec = (np.array(y_pred) - np.array(y_true)) / np.array(y_true)
			else:
				error_vec = np.abs(np.array(y_pred) - np.array(y_true))

			plt.clf()
			plt.scatter(y_true, error_vec, s = 1)
			plt.savefig(f"{direc}/{model_name}-{label}-Hetero")

			# To add the Y column for the hetero analysis
			data = x.copy()
			data['Y'] = y_true

			file_counter = 0
			first = True
			for i, col in enumerate(data.columns):

				# Creating the figs and axes
				if first:
					fig, ax = plt.subplots(nrows=3, ncols=3)
					fig.tight_layout()
					first = False

				counter = i % 9
				row_idx = int (counter/3)
				col_idx = counter % 3

				ax[row_idx, col_idx].set_title(col)
				ax[row_idx, col_idx].scatter(data[col], error_vec, s = 1)

				if (i % 9 == 8) or (i == len(data.columns)-1):
					
					# Unless for the first time, files shoud be saved
					plt.savefig(f"{direc}/{model_name}-{label}-Hetero-{file_counter}")
					plt.close()

					file_counter += 1

					if i != len(data.columns)-1:
						fig, ax = plt.subplots(nrows=3, ncols=3)
						fig.tight_layout()
						
		except ZeroDivisionError:
			print ("Unable to plot heteroskedasticity graphs. Output variable contains zero")