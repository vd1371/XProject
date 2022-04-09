import numpy as np
import pandas as pd

def mix_and_split(x_train, x_cv, y_train, y_cv):

	if isinstance(x_train, pd.DataFrame): x_train = x_train.values
	if isinstance(x_cv, pd.DataFrame): x_cv = x_cv.values

	x = np.array(x_train.tolist() + x_cv.tolist())
	y = np.array(y_train.tolist() + y_cv.tolist())

	l = len(x)
	indices = np.random.choice(l, len(y_train))

	x_train = x[indices]
	y_train = y[indices]

	mask = np.ones(l, dtype=bool)
	mask[indices] = False

	x_cv = x[mask]
	y_cv = y[mask]

	return x_train, x_cv, y_train, y_cv

