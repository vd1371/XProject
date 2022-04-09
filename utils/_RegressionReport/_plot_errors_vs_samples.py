
import matplotlib.pyplot as plt
from ._get_direc_for_label import get_direc_for_label

def plot_errors_vs_samples(report, label, **params):

	direc = params.get('direc')
	direc = get_direc_for_label(direc, label)
	model_name = params.get('model_name')

	# Plotting errors
	errs = list(report['Error'])
	ticks = [i for i in range(len(errs))]
	plt.clf()
	plt.ylabel('Erros')
	plt.title(label+'-Errors')
	plt.scatter(ticks, errs, s = 1)
	plt.grid(True)
	plt.savefig(f"{direc}/{model_name}-{label}-Errors.png")
	plt.close()

