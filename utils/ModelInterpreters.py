import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pprint

import shap
import lime
import lime.lime_tabular

from utils.FeatureImportanceReport import report_feature_importance

def shap_deep_regression(direc, model, x_train, x_test, cols, num_top_features = 10, logger = None, label = 'DNN-OnTest'):

	explainer = shap.DeepExplainer(model, x_train.values)
	shap_values = explainer.shap_values(x_test.values)

	shap.summary_plot(shap_values[0], features = x_test, feature_names=cols, show=False)
	plt.tight_layout()
	plt.savefig(direc + f"/ShapValues-{label}.png")
	plt.close()
	
	shap_values = pd.DataFrame(shap_values[0], columns = list(x_train.columns)).abs().mean(axis = 0)
	logger.info(f"SHAP Values {label}\n" + pprint.pformat(shap_values.nlargest(num_top_features)))
	
	ax = shap_values.nlargest(num_top_features).plot(kind='bar', title = label)
	fig = ax.get_figure()
	

	plt.tight_layout()
	fig.savefig(direc + "/"+ f'ShapValuesBar-{label}.png')
	del fig
	plt.close()
		


def FIIL(direc, model, eval_method, x, y, n_top_features, n_simulations = 10, logger = None, label = ""):
		
	base_error = eval_method(model.predict(x), y)
	print (f'Model Error:{base_error:.2f}')
	
	feature_importances_ = []
	for col in x.columns:
		x_temp = x.copy()
		temp_err = []
		for _ in range(n_simulations):
			np.random.shuffle(x_temp[col].values)
			err = base_error - eval_method(model.predict(x_temp), y)
			temp_err.append(err)
		feature_importances_.append(abs(round(np.mean(temp_err), 4)) if np.mean(temp_err)<0 else 0)
	
	report_feature_importance(direc, np.array(feature_importances_), x.columns, n_top_features, label, logger)