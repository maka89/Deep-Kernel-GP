import numpy
from .optimizers import Adam
from .utils import grad_check,calc_acc,one_hot

from .layers import Activation,Dense,Dropout,CovMat
from .loss import mse_loss,cce_loss

from scipy.linalg import cholesky,cho_solve,solve_triangular

	
	
class CoreNN:
	#Hidden layers - list of layers.
	#costfn - costfunction in the form as in loss.py
	def __init__(self,layers,costfn):
		self.layers=layers
		self.cost=costfn
		
		
	def forward(self,X,gc=False):
		
		A=X
		if not gc:
			for i in range(0,len(self.layers)):
				A=self.layers[i].forward(A)
		else:
			for i in range(0,len(self.layers)):
				A=self.layers[i].predict(A)

		return A
	
	def backward(self,Y):
		self.j,err=self.cost(Y,self.layers[-1].out)
		for i in reversed(range(0,len(self.layers))):
			err=self.layers[i].backward(err)
		return err
	
	#First run of NN. calculate inp shapes of layers, initialize weights, add activation layers and ouput layer.
	def first_run(self,X,Y):
		A=X
		if not self.layers:
			brflag=True
		else:
			brflag=False
		i=0
		while not brflag:
			if type( self.layers[i] ) == int:
				self.layers[i]=Dense(self.layers[i],activation='tanh')
				
			self.layers[i].set_inp(A.shape[-1])
			if self.layers[i].trainable:
				self.layers[i].initialize_ws()
				if self.layers[i].activation is not None:
					self.layers.insert(i+1,Activation(self.layers[i].activation))
			A=self.layers[i].forward(A)
			i+=1
			if i==len(self.layers):
				brflag=True
				
		for i in range(0,len(self.layers)):
			if type(self.layers[i]) != Dropout and type(self.layers[i]) != CovMat:
				self.layers[i].predict=self.layers[i].forward
		
		
	def grad_check(self,X,Y,n_checks=100):
		return grad_check(self,X,Y,n_checks)

				
class NNRegressor(CoreNN):
	def __init__(self,layers=[64],opt=None,maxiter=200,batch_size=64,gp=True,verbose=False):
		super().__init__(layers,mse_loss)
		if gp:
			self.cost=self.gp_loss
		self.opt=opt
		self.verbose=verbose
		self.maxiter=maxiter
		self.batch_size=batch_size
		self.fitted=False
		self.opt=opt
		self.task=0
		
	def gp_loss(self,y,K):
		self.y=y
		self.A=self.layers[-2].out
		self.K=K
		self.L_ = cholesky(K, lower=True)
		
		L_inv = solve_triangular(self.L_.T,numpy.eye(self.L_.shape[0]))
		self.K_inv = L_inv.dot(L_inv.T)
		
		self.alpha_ = cho_solve((self.L_, True), y)
		self.nlml=0.0
		self.nlml_grad=0.0
		for i in range(0,y.shape[1]):
			
			gg1=numpy.dot(self.alpha_[:,i].reshape(1,-1),y[:,i].reshape(-1,1))[0,0]

			self.nlml+=0.5*gg1+numpy.sum(numpy.log(numpy.diag(self.L_)))+K.shape[0]*0.5*numpy.log(2.0*numpy.pi)
			yy=numpy.dot(y[:,i].reshape(-1,1),y[:,i].reshape(1,-1))
			self.nlml_grad += -0.5*( numpy.dot(numpy.dot(self.K_inv,yy),self.K_inv)-self.K_inv)*K.shape[0]

		return self.nlml,self.nlml_grad
	def fast_forward(self,X):
		A=X
		for i in range(0,len(self.layers)-1):
			A=self.layers[i].predict(A)
		return A
	def fit(self,X,Y,batch_size=None,maxiter=None):
		if batch_size is not None:
			self.batch_size=batch_size
		if maxiter is not None:
			self.maxiter=maxiter
		if self.opt is None:
			self.opt=Adam()
		if not self.fitted:
			self.first_run(X[0:2],Y[0:2])
			
		
		a=self.opt.fit(X,Y,self,batch_size=self.batch_size,maxiter=self.maxiter,verbose=self.verbose)

		self.fitted=True
		
		self.y=Y
		self.x=X

		return a
	def predict(self,X):
		A=X
		A2=self.x
		for i in range(0,len(self.layers)-1):
			A2=self.layers[i].predict(A2)
			A=self.layers[i].predict(A)
			
		self.K=self.layers[-1].forward(A2)
		self.L_ = cholesky(self.K, lower=True)
		
		L_inv = solve_triangular(self.L_.T,numpy.eye(self.L_.shape[0]))
		self.K_inv = L_inv.dot(L_inv.T)
		
		self.alpha_ = cho_solve((self.L_, True), self.y)
		
		
		K2=numpy.zeros((X.shape[0],X.shape[0]))
		K3=numpy.zeros((X.shape[0],self.K.shape[0]))
		
		if self.layers[-1].kernel=='rbf':
			d1=0.0
			d2=0.0
			for i in range(0,A.shape[1]):
				d1+=(A[:,i].reshape(-1,1)-A[:,i].reshape(1,-1))**2
				d2+=(A[:,i].reshape(-1,1)-A2[:,i].reshape(1,-1))**2
			K2=self.layers[-1].var*numpy.exp(-0.5*d1)+numpy.identity(A.shape[0])*(self.layers[-1].s_alpha+1e-8)
			K3=self.layers[-1].var*numpy.exp(-0.5*d2)
		elif self.layers[-1].kernel=='dot':
			K2=numpy.dot(A,A.T)+numpy.identity(A.shape[0])*(self.layers[-1].s_alpha+1e-8) + self.layers[-1].var
			K3=numpy.dot(A,A2.T) + self.layers[-1].var
			
		preds=numpy.zeros((X.shape[0],self.y.shape[1]))
		for i in range(0,self.alpha_.shape[1]):
			preds[:,i]=numpy.dot(K3,self.alpha_[:,i].reshape(-1,1))[:,0]
		
		return preds, numpy.sqrt(numpy.diagonal(K2-numpy.dot(K3,numpy.dot(self.K_inv,K3.T))))
		
		
	def update(self,X,Y):
		self.forward(X)
		self.backward(Y)
		return self.layers[-1].out
