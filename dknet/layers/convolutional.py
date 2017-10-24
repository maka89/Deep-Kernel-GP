import numpy
from numpy import unravel_index
from .activation import Activation
from .layer import Layer
		
class Conv2D(Layer):
	def __init__(self,n_out,kernel_size,activation=None):
		self.n_out=n_out
		self.activation=activation
		self.kernel_size=kernel_size
		self.trainable=True
	def initialize_ws(self):
		self.W=numpy.random.randn(self.kernel_size[0],self.kernel_size[1],self.n_inp,self.n_out).astype(dtype=self.dtype)*numpy.sqrt(1.0/(self.n_inp*numpy.prod(self.kernel_size)))
		self.b=numpy.zeros((1,self.n_out),dtype=self.dtype)
		self.dW=numpy.zeros_like(self.W,dtype=self.dtype)
		self.db=numpy.zeros_like(self.b,dtype=self.dtype)
		assert(self.W.shape[0]%2!=0) #Odd filter size pls
		assert(self.W.shape[1]%2!=0) #Odd fiter size pls
	def forward(self,X):
		self.inp=X
		
		hpad,wpad=int(self.W.shape[0]/2),int(self.W.shape[1]/2)
		X2=numpy.zeros((X.shape[0],X.shape[1]+2*hpad,X.shape[2]+2*wpad,X.shape[3]),dtype=self.dtype)
		X2[:,hpad:X2.shape[1]-hpad,wpad:X2.shape[2]-wpad,:]=numpy.copy(X)
		A=numpy.zeros((X.shape[0],X.shape[1],X.shape[2],self.n_out),dtype=self.dtype)
		M,N=X.shape[1],X.shape[2]
		for i in range(0,M):
			for j in range(0,N):
				A[:,i,j,:]=numpy.sum(X2[:,hpad+i-hpad:hpad+i+hpad+1,wpad+j-wpad:wpad+j+wpad+1,:][:,:,:,:,numpy.newaxis]*self.W[numpy.newaxis,:,:,:,:],axis=(1,2,3))
		A+=self.b[0,:]
		
		self.out=A
		return self.out
	
	def backward(self,err):
			
		X=self.inp
		hpad,wpad=int(self.W.shape[0]/2),int(self.W.shape[1]/2)
		X2=numpy.zeros((X.shape[0],X.shape[1]+2*hpad,X.shape[2]+2*wpad,X.shape[3]),dtype=self.dtype)
		X2[:,hpad:X2.shape[1]-hpad,wpad:X2.shape[2]-wpad,:]=numpy.copy(X)
		
		tmpdW=numpy.zeros_like(self.dW,dtype=self.dtype)
		dodi=numpy.zeros_like(X2,dtype=self.dtype)
		M,N=X.shape[1],X.shape[2]
		for i in range(0,M):
			for j in range(0,N):
				tmpdW+=numpy.sum(err[:,i,j,:][:,numpy.newaxis,numpy.newaxis,numpy.newaxis,:]*X2[:,i:i+2*hpad+1,j:j+2*wpad+1,:][:,:,:,:,numpy.newaxis],0)
				dodi[:,i:i+2*hpad+1,j:j+2*wpad+1,:]+=numpy.sum(err[:,i,j,:][:,numpy.newaxis,numpy.newaxis,numpy.newaxis,:]*self.W[numpy.newaxis,:,:,:,:],-1)
		self.dW=tmpdW
		self.db[0,:]=numpy.sum(err,(0,1,2))
		
		return dodi[:,hpad:dodi.shape[1]-hpad,wpad:dodi.shape[2]-wpad,:]