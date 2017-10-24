import numpy
from numpy import unravel_index
from .layer import Layer
class MaxPool2D(Layer):
	def __init__(self,pool_size=(2,2)):
		self.trainable=False
		self.pool_size=pool_size
	def forward(self,X):
		self.inp=X
		self.mask=numpy.zeros_like(X)
		assert(X.shape[1]%self.pool_size[0]==0)
		assert(X.shape[2]%self.pool_size[1]==0)
		self.out=numpy.zeros((X.shape[0],int(X.shape[1]/self.pool_size[0]),int(X.shape[2]/self.pool_size[1]),X.shape[3]))
		
		for i in range(0,self.out.shape[1]):
			for j in range(0,self.out.shape[2]):
				a=X[:,self.pool_size[0]*i:self.pool_size[0]*(i+1),self.pool_size[1]*j:self.pool_size[1]*(j+1),:]
				mv=numpy.max(a,axis=(1,2))
				self.out[:,i,j,:] = mv
				self.mask[:,self.pool_size[0]*i:self.pool_size[0]*(i+1),self.pool_size[1]*j:self.pool_size[1]*(j+1),:]=mv[:,numpy.newaxis,numpy.newaxis,:]
		return self.out
		
	def backward(self,err):
		err2=numpy.zeros_like(self.inp)
		for i in range(0,self.out.shape[1]):
			for j in range(0,self.out.shape[2]):
				mm=(self.mask[:,self.pool_size[0]*i:self.pool_size[0]*(i+1),self.pool_size[1]*j:self.pool_size[1]*(j+1),:]==self.inp[:,self.pool_size[0]*i:self.pool_size[0]*(i+1),self.pool_size[1]*j:self.pool_size[1]*(j+1),:])
				ms=numpy.sum(mm,axis=(1,2))
				err2[:,self.pool_size[0]*i:self.pool_size[0]*(i+1),self.pool_size[1]*j:self.pool_size[1]*(j+1),:]=(mm/ms[:,numpy.newaxis,numpy.newaxis,:])*err[:,i,j,:][:,numpy.newaxis,numpy.newaxis,:]
		return err2
class AveragePool2D(Layer):
	def __init__(self,pool_size=(2,2)):
		self.trainable=False
		self.pool_size=pool_size
	def forward(self,X):
		self.inp=X
		assert(X.shape[1]%self.pool_size[0]==0)
		assert(X.shape[2]%self.pool_size[1]==0)
		self.out=numpy.zeros((X.shape[0],int(X.shape[1]/self.pool_size[0]),int(X.shape[2]/self.pool_size[1]),X.shape[3]))
		
		for i in range(0,self.out.shape[1]):
			for j in range(0,self.out.shape[2]):
				a=X[:,self.pool_size[0]*i:self.pool_size[0]*(i+1),self.pool_size[1]*j:self.pool_size[1]*(j+1),:]
				mv=numpy.average(a,axis=(1,2))
				self.out[:,i,j,:] = mv
		return self.out
		
	def backward(self,err):
		err2=numpy.zeros_like(self.inp)
		for i in range(0,self.out.shape[1]):
			for j in range(0,self.out.shape[2]):
				err2[:,self.pool_size[0]*i:self.pool_size[0]*(i+1),self.pool_size[1]*j:self.pool_size[1]*(j+1),:]=err[:,i,j,:][:,numpy.newaxis,numpy.newaxis,:]/numpy.prod(self.pool_size)
		return err2