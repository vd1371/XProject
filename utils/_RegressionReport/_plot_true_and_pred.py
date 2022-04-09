import matplotlib.pyplot as plt
from ._get_direc_for_label import get_direc_for_label

def plot_true_and_pred(y_true, y_pred, label, **params):

	direc = params.get('direc')
	direc = get_direc_for_label(direc, label)
	model_name = params.get('model_name')

	# Actual and Predicted Plotting
	plt.clf()
	plt.xlabel('Sample')
	plt.ylabel('Value')
	plt.title(label + "-True and Predicted")
	ticks = [i for i in range(len(y_true))]
	act = plt.plot(ticks, y_true, label = "True Vals")
	pred = plt.plot (ticks, y_pred, label = 'Predicted Vals')
	plt.legend()
	plt.grid(True)
	plt.savefig(f"{direc}/{model_name}-{label}-TrueAndPreds.png")
	plt.clf()
	plt.close()