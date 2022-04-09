import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import utils

def get_callbacks(**params):

	min_delta = params.get('min_delta')
	patience = params.get("patience")
	should_early_stop = params.get("should_early_stop")
	should_plot_live_error = params.get("should_plot_live_error")
	should_checkpoint = params.get("should_checkpoint")
	directory = params.get("directory")
	file_name = params.get("dl").file_name
	n_voters = params.get("n_voters")

	all_callbacks = []
	for i in range(n_voters):

		callback_list = []
		early_stopping = EarlyStopping(monitor='loss',
										min_delta = min_delta,
										patience=patience,
										verbose=1,
										mode='auto') 
		plot_losses = utils.PlotLosses()

		if should_early_stop:
			callback_list.append(early_stopping)
		if should_plot_live_error:
			callback_list.append(plot_losses)
		if should_checkpoint:
			checkpoint = ModelCheckpoint(os.path.join(directory,
													f'{file_name}-BestModel-Voter{i}.h5'),
													monitor='val_loss',
													verbose=1,
													save_best_only=True,
													mode='auto')
			callback_list.append(checkpoint)

		all_callbacks.append(callback_list)

	return all_callbacks