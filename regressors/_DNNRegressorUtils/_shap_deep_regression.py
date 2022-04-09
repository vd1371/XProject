import pandas as pd
import shap
import matplotlib.pyplot as plt
import pprint

def shap_deep_regression(direc, model, x_train, x_test, cols, num_top_features = 10, logger = None, label = 'DNN-OnTest'):

	explainer = shap.DeepExplainer(model, x_train.values)
	shap_values = explainer.shap_values(x_test.values)

	shap.summary_plot(shap_values[0], features = x_test, feature_names=cols, show=False)
	plt.tight_layout()
	plt.savefig(direc + f"/ShapValues-{label}.png")
	plt.close()
	
	shap_values = pd.DataFrame(shap_values[0], columns = list(x_train.columns)).abs().mean(axis = 0)
	logger.info(f"SHAP Values {label}\n" + pprint.pformat(shap_values.nlargest(num_top_features)))
	


	ax = shap_values.nlargest(min(num_top_features, len(cols))).plot(kind='bar', title = label)
	fig = ax.get_figure()
	

	plt.tight_layout()
	fig.savefig(direc + "/"+ f'ShapValuesBar-{label}.png')
	del fig
	plt.close()