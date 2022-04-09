import pandas as pd

def save_history_of_model(hist, i, **params):

	directory = params.get("directory")
	loss_func = params.get("loss_func")

	hist_df = pd.DataFrame.from_dict(hist.history)
	hist_df.to_csv(f"{directory}/{loss_func}-hist-Voter{i}.csv")
	

