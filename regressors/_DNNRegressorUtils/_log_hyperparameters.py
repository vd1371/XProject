import pprint

def log_hyperparameters(**params):

		dl = params.get('dl')
		n_voters = params.pop('n_voters')
		layers = params.pop("layers")
		input_activation_func = params.pop("input_activation_func")
		hidden_activation_func = params.pop("hidden_activation_func")
		final_activation_func = params.pop("final_activation_func")
		loss_func = params.pop("loss_func")
		epochs = params.pop("epochs")
		min_delta = params.pop("min_delta")
		patience = params.pop("patience")
		batch_size = params.pop('batch_size')
		should_early_stop = params.pop("should_early_stop")
		should_plot_live_error = params.pop("should_plot_live_error")
		should_checkpoint = params.pop("should_checkpoint")
		regul_type = params.pop("regul_type")
		reg_param = params.pop("reg_param")
		optimizer = params.pop("optimizer")

		dl.log.info(pprint.pformat({'n_voters': n_voters,
									'layers': layers,
									'input_activation_func': input_activation_func,
									'hidden_activation_func': hidden_activation_func,
									'final_activation_func': final_activation_func,
									'loss_func': loss_func,
									'epochs': epochs,
									'min_delta': min_delta,
									'patience': patience,
									'batch_size': batch_size,
									'should_early_stop': should_early_stop,
									'regularization_type': regul_type,
									'reg_param': reg_param,
									'random_state': dl.random_state}))