import numpy
def mse_loss(y_true,y_pred):
	return 0.5*numpy.average((y_true-y_pred)**2),(y_pred-y_true)/numpy.prod(y_true.shape)
	
def cce_loss(y_true,y_pred):
	return -numpy.average(numpy.sum(y_true*numpy.log(y_pred),1)), (y_pred-y_true)/(y_pred*(1.0-y_pred)+1e-12)/y_true.shape[0]