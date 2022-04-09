import os

def get_direc_for_label(direc, label):

	direc_for_label = f"{direc}/{label}"

	if not os.path.exists(direc_for_label):
		os.makedirs(direc_for_label)

	return direc_for_label

