

def close_live_plot(callback_list):

	# Closing the plot losses
	try:
		for callback in callback_list:
			callback.closePlot()
	except:
		pass