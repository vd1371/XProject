
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

import utils
from ._log_crossval_results import log_crossval_results

def cross_validate_model(model, model_name, **params):

	cv_numbers = params.get("k")
	X = params.get("X")
	Y = params.get("Y")

	r2_scorer = make_scorer(utils.R2, greater_is_better=False)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    scores = cross_validate(model, X, Y, 
                            cv=cv_numbers,
                            verbose=0,
                            scoring= {'MSE': mse_scorer, 'R2' : r2_scorer})

    log_crossval_results(model_name, scores, self.log)