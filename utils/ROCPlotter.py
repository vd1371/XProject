#Imporing dependencies
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def plot_roc(y_test_dummies, y_pred_probs, classes, direc):
	'''Plotting ROC curve
	
	This function is based on the assumption that y_test_dummies and
	y_pred_probs are np.ndarrays. y_test_dummies is one-hot-encoded 
	y_test without dropping the firest. y_pred_probs is also a 2D array
	with each dimenstion corresponding to each binarized class in y_test_dummies
	'''
	fpr = {}
	tpr = {}
	roc_auc = {}

	for i in range(len(classes)):

		fpr[i], tpr[i], _ = roc_curve(y_test_dummies[i], y_pred_probs[i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Plot of a ROC curve for a specific class
	for i, label in enumerate(classes):
		plt.figure(figsize = (3, 2))
		plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[label]:.2f})')
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend(loc="best")
		plt.savefig(direc + f"/ROC{label}.tiff", dpi = 300, bbox_inches = 'tight')

		plt.clf()
		plt.close("all")