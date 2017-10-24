import numpy
from numpy import unravel_index
from .layer import Layer
class Flatten(Layer):
	def __init__(self):
		self.trainable=False
	def forward(self,X):
		self.inp=numpy.copy(X)
		self.out=X.reshape(X.shape[0],numpy.prod(X.shape[1::]))
		return self.out
	def backward(self,err):
		return err.reshape(self.inp.shape)