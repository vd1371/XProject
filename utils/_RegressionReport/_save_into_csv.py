import pandas as pd
from ._get_direc_for_label import get_direc_for_label

def save_into_csv(y_true, y_pred, inds, label, **params):

	direc = params.get('direc')
	direc = get_direc_for_label(direc, label)
	model_name = params.get('model_name')

	# Saving into csv file
	report = pd.DataFrame()
	report['Actual'] = y_true
	report['Predicted'] = y_pred
	report['Error'] = report['Predicted'] - report['Actual']
	report['ErrorPercent'] = (report['Predicted'] - report['Actual'])/report['Actual']
	report['Ind'] = inds
	report.set_index('Ind', inplace=True)
	report.to_csv(direc + "/"+ f'{model_name}-{label}.csv')

	return report

