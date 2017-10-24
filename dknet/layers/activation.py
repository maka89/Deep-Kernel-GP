import numpy
from numpy import unravel_index
from .layer import Layer
def relu(x,dtype=numpy.float64):
	tmp=(x>=0)
	return x*tmp,1*tmp

def sigmoid(x,dtype=numpy.float64):
	a=1.0/(numpy.exp(-x)+1.0)
	return a, a*(1-a)
	
def linear(x,dtype=numpy.float64):
	return x,1.0#numpy.ones_like(x,dtype=dtype)
	
def tanh(x,dtype=numpy.float64):
	a=numpy.tanh(x)
	return a, 1.0-a**2
	
def lrelu(x,dtype=numpy.float64):
	y=(x>=0)*1.0+(x<0)*0.01
	return y*x,y
	
def softplus(x,dtype=numpy.float64):
	tmp=numpy.exp(x)
	return numpy.log(tmp+1.0), tmp/(1.0+tmp)
	
def softmax(x,dtype=numpy.float64):
	s=numpy.exp(x)
	s=s/numpy.sum(s,1)[:,numpy.newaxis]
	return s,s*(1.0-s)

def rbf(x,dtype=numpy.float64):
	
	s=numpy.exp(-0.5*numpy.sum(x**2,-1))
	print(x.shape,s.shape)
	return s, -x*s[:,:,numpy.newaxis]

class Activation(Layer):

	dict={'linear':linear,'relu':relu,'sigmoid':sigmoid,'tanh':tanh,'softmax':softmax,'lrelu':lrelu,'softplus':softplus,'rbf':rbf}	
			
	def __init__(self,strr):
		
		if strr in self.dict.keys():
			self.afstr=strr
			self.af=self.dict[strr]
		else:
			print("Error. Undefined activation function '" + str(strr)+"'. Using linear activation.")
			print("Available activations: " + str(list(self.dict.keys())))
			self.af=linear
			self.afstr='linear'
		self.trainable=False
	def forward(self,X):
		self.inp=X
		self.a=self.af(X,dtype=self.dtype)
		self.out=self.a[0]
		return self.out
	def backward(self,err):
		return self.a[1]*err
		
