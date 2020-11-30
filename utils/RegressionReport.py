from utils.BaseModel import R2, MAPE, CorCoef

import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score

import shap
import lime, lime.lime_tabular


def evaluate_regression(direc, x_true,
								y_true,
								y_pred,
								inds,
								label,
								logger,
								slicer = 1,
								should_check_hetero = True,
								should_log_inverse = False):

	'''
	x_true MUST be panda dataframe
	y_true and y_pred must be list or 1d numpy array
	'''

	if should_log_inverse:
		y_true = np.exp(y_true)
		y_pred = np.exp(y_pred)
	
	# Saviong into csv file
	report = pd.DataFrame()
	report['Actual'] = y_true
	report['Predicted'] = y_pred
	report['Error'] = report['Actual'] - report['Predicted']
	report['Ind'] = inds
	report.set_index('Ind', inplace=True)
	report.to_csv(direc + "/"+ f'{label}.csv')
	
	report_str = f"{label}, CorCoef= {CorCoef(y_true, y_pred):.2f}, "\
					f"R2= {R2(y_true, y_pred):.2f}, RMSE={MSE(y_true, y_pred)**0.5:.2f}, "\
						f"MSE={MSE(y_true, y_pred):.2f}, MAE={MAE(y_true, y_pred):.2f}, "\
							f"MAPE={MAPE(y_true, list(y_pred)):.2f}%"
	
	logger.info(report_str)
	print(report_str)
	
	# Plotting errors
	errs = list(report['Error'])
	x = [i for i in range(len(errs))]
	plt.clf()
	plt.ylabel('Erros')
	plt.title(label+'-Errors')
	plt.scatter(x, errs, s = 1)
	plt.grid(True)
	plt.savefig(direc + '/' + label + '-Errors.png')
	plt.close()

	if slicer != 1:
		y_true, y_pred = y_true[-int(slicer*len(y_true)):], y_pred[-int(slicer*len(y_pred)):]
	
	# let's order them
	temp_list = []
	for true, pred in zip(y_true, y_pred):
		temp_list.append([true, pred])
	temp_list = sorted(temp_list , key=lambda x: x[0])
	y_true, y_pred = [], []
	for i, pair in enumerate(temp_list):
		y_true.append(pair[0])
		y_pred.append(pair[1])
		
	# Actual vs Predicted plotting
	plt.clf()
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.title(label+'-Actual vs. Predicted')
	ac_vs_pre = plt.scatter(y_true, y_pred, s = 1)
	plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=0.75)
	plt.grid(True)
	plt.savefig(direc + '/' + label + '-ACvsPRE.png')
	plt.close()
	
	# Actual and Predicted Plotting
	plt.clf()
	plt.xlabel('Sample')
	plt.ylabel('Value')
	plt.title(label + "-Actual and Predicted")
	x = [i for i in range(len(y_true))]
	act = plt.plot(x, y_true, label = "actual")
	pred = plt.plot (x, y_pred, label = 'predicted')
	plt.legend()
	plt.grid(True)
	plt.savefig(direc + '/' + label + '-ACandPRE.png')
	plt.clf()
	plt.close()

	# Plotting the error percentage vs. feature and the output variable

	if should_check_hetero:
		try:

			mape_vec = (np.array(y_pred) - np.array(y_true)) / np.array(y_true)

			plt.clf()
			plt.scatter(y_true, mape_vec, s = 1)
			plt.savefig(f"{direc}/{label}-Hetero")

			data = x_true.copy()
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
				ax[row_idx, col_idx].scatter(data[col], mape_vec, s = 1)

				if (i % 9 == 8) or (i == len(data.columns)-1):
					
					# Unless for the first time, files shoud be saved
					plt.savefig(f"{direc}/{label}-Hetero-{file_counter}")
					plt.close()

					file_counter += 1

					if i != len(data.columns)-1:
						fig, ax = plt.subplots(nrows=3, ncols=3)
						fig.tight_layout()
						
		except ZeroDivisionError:
			print ("Unable to plot heteroskedasticity graphs. Output variable contains zero")
