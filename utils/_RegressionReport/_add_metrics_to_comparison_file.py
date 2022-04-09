import os
import pandas as pd

def add_metrics_to_comparison_file(metrics, label, **params):

	direc = params.get('direc')
	model_name = params.get('model_name')

	# Comparing the results of all models
	compare_direc = os.path.abspath(direc + "/../") + "/Comparision.csv"
	if os.path.exists(compare_direc):
		compar_df = pd.read_csv(compare_direc, index_col = 0)
	else:
		compar_df = pd.DataFrame(columns = list(metrics.keys()))

	compar_df.loc[model_name + "-" + label] = list(metrics.values())
	compar_df.to_csv(compare_direc)