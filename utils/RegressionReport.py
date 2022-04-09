import os
import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt

from ._RegressionReport import *

def evaluate_regression(*args, **params):
	'''Report classification results
	
	*args should be lists of [label, x, y, inds]

	x MUST be panda dataframe
	y_true and y_pred must be list or 1d numpy array
	'''
	direc = params.get('direc')
	model = params.get('model')
	model_name = params.get('model_name')
	logger = params.get('logger')
	slicer = params.get('slicer', 1)

	for ls in args:

		label, x, y_true, y_pred, inds = get_x_y_true_pred(ls, **params)

		metrics = log_metrics(y_true, y_pred, label, **params)
		add_metrics_to_comparison_file(metrics, label, **params)

		x, y_true, y_pred, inds = slice_x_y(x, y_true, y_pred, inds, **params)
		report = save_into_csv(y_true, y_pred, inds, label, **params)
		plot_errors_vs_samples(report, label, **params)
	
		y_true_for_plot, y_pred_for_plot = sort_ys_based_on_y_true(y_true, y_pred)
		plot_true_vs_pred(y_true_for_plot, y_pred_for_plot, label, **params)
		plot_true_and_pred(y_true_for_plot, y_pred_for_plot, label, **params)

		plot_hetero(x, y_true, y_pred, label, **params)
