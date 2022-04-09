from ..R2 import R2
from ..MAPE import MAPE
from ..CorCoef import CorCoef

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

def log_metrics(y_true, y_pred, label, **params):

	direc = params.get('direc')
	model = params.get('model')
	model_name = params.get('model_name')
	logger = params.get('logger')

	corcoef_ = CorCoef(y_true, y_pred)
	r2_ = R2(y_true, y_pred)
	mse_ = MSE(y_true, y_pred)
	mae_ = MAE(y_true, y_pred)
	mape_ = MAPE(y_true, list(y_pred))

	# Reporting the quantitative results
	report_str = f"{model_name}-{label}, CorCoef= {corcoef_:.2f}, "\
					f"R2= {r2_:.2f}, RMSE={mse_**0.5:.2f}, "\
						f"MSE={mse_:.2f}, MAE={mae_:.2f}, "\
							f"MAPE={mape_:.2f}%"
	
	logger.info(report_str)
	print(report_str)

	return {'corcoef_': corcoef_,
			'r2_': r2_,
			'mse_': mse_,
			'mae_': mae_,
			'mape_': mape_}

