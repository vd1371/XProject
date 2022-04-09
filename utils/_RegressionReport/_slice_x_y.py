

def slice_x_y(x, y_true, y_pred, inds, **params):

	slicer = params.get("slicer")

	if slicer != 1:
		y_true = y_true[-int(slicer*len(y_true)):]
		y_pred = y_pred[-int(slicer*len(y_pred)):]
		x = x[-int(slicer*len(x)):]
		inds = inds[-int(slicer*len(inds)):]

	return x, y_true, y_pred, inds