import numpy
from .layer import Layer
class Dropout(Layer):

	def __init__(self,keep_prob):
		self.keep_prob=keep_prob
		self.trainable=False
	def forward(self,X):
		self.inp=X
		self.mask=(numpy.random.random(size=X.shape).astype(self.dtype)<=self.keep_prob)/self.keep_prob
		self.out=self.mask*self.inp
		return self.out
	
	def predict(self,X):
		return X
	
	def backward(self,err):
		return err*self.mask