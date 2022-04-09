

def log_crossval_results(model_name, scores, logger):

	to_be_logged = f"|- Cross validation is done for {model_name} "\
					f"RMSE: {(-np.mean(scores['test_MSE']))**0.5:.2f},"\
						f"MSE: {-np.mean(scores['test_MSE']):.2f}, "\
	                    	f"R2: {-np.mean(scores['test_R2']):.2f} -|"
	
	logger.info(to_be_logged)
	print(to_be_logged)