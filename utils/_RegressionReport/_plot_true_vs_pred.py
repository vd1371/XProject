import matplotlib.pyplot as plt
from ._get_direc_for_label import get_direc_for_label

def plot_true_vs_pred(y_true, y_pred, label, **params):

	direc = params.get('direc')
	direc = get_direc_for_label(direc, label)
	model_name = params.get('model_name')

	# Actual vs Predicted plotting
	plt.clf()
	plt.xlabel('True')
	plt.ylabel('Predictions')
	plt.title(label+'-True vs. Predicted')
	ac_vs_pre = plt.scatter(y_true, y_pred, s = 1)
	plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=0.75)
	plt.grid(True)
	plt.savefig(f"{direc}/{model_name}-{label}-TrueVsPred.png")
	plt.close()