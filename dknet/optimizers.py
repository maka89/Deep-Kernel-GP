import numpy
from .utils import r2,calc_acc
from scipy.linalg import eigh
import time
class Optimizer:
		
	def weight_grads_as_arr(self):
		x=numpy.zeros((0,))
		for i in range(0,len(self.model.layers)):
			if self.model.layers[i].trainable:
				x=numpy.concatenate((x,self.model.layers[i].dW.ravel()))
				x=numpy.concatenate((x,self.model.layers[i].db.ravel()))
		return x
	def weight_grads_std_as_arr(self):
		x=numpy.zeros((0,))
		for i in range(0,len(self.model.layers)):
			if self.model.layers[i].trainable:
				x=numpy.concatenate((x,self.model.layers[i].dWs.ravel()))
				x=numpy.concatenate((x,self.model.layers[i].dbs.ravel()))
		return x
	def weights_as_arr(self):
		x=numpy.zeros((0,))
		for i in range(0,len(self.model.layers)):
			if self.model.layers[i].trainable:
				x=numpy.concatenate((x,self.model.layers[i].W.ravel()))
				x=numpy.concatenate((x,self.model.layers[i].b.ravel()))
		return x
		
	def update_params_from_1darr(self,x):
		n=0
		for i in range(0,len(self.model.layers)):
			if self.model.layers[i].trainable:
				shape = self.model.layers[i].W.shape
				mm=numpy.prod(shape)
				self.model.layers[i].W=x[n:n+mm].reshape(shape)
				n+=mm
				
				shape = self.model.layers[i].b.shape
				mm=numpy.prod(shape)
				self.model.layers[i].b=x[n:n+mm].reshape(shape)
				n+=mm

class ALR(Optimizer):
	def __init__(self,learning_rate=1e-3,beta=0.99,C=500):
		super()
		self.learning_rate=learning_rate
		self.dlr=numpy.zeros_like(self.learning_rate)
		self.beta=beta
		self.C=C
		self.first_run=True
	def reset(self):
		self.init_moments()
	def init_moments(self):
		self.m11=numpy.zeros_like(self.weights_as_arr())
		self.m1=numpy.zeros_like(self.weights_as_arr())
		self.m2=numpy.zeros_like(self.weights_as_arr())		
		self.t=0
		#self.learning_rate*=numpy.ones_like(self.weights_as_arr())
		
	def fit(self,X,Y,model,batch_size=16,maxiter=100,verbose=True):
		self.model=model
		self.n_iter=0
		#self.beta=numpy.exp(-batch_size/self.C)
		if self.first_run:
			self.init_moments()
			self.first_run=False
		brflag=False
		self.save=[]
		full_batch=X.shape[0]
		m=int(X.shape[0]/batch_size)
		j=0
		for i in range(0,100000):
			
			
			
			x=self.weights_as_arr()
			
			
			num_mb=0.0
			dw=0.0
			dws=0.0
			score=0.0
			batch_done=False
			while not batch_done:
				
				
				batch_x=X[j*batch_size:(j+1)*batch_size]
				batch_y=Y[j*batch_size:(j+1)*batch_size]
				
				#Calc raw gradients
				self.model.update(batch_x,batch_y)
				
				wtmp=self.weight_grads_as_arr()
				dw=(num_mb*dw+wtmp)/(num_mb+1)
				stmp=self.weight_grads_std_as_arr()-wtmp**2
				dws=(num_mb*dws+stmp)/(num_mb+1)
				score=(score*num_mb+model.j)/(num_mb+1)
				num_mb+=1
				
				if not numpy.all(numpy.isfinite(dws)) or numpy.any(dws<0):
					print("ERROR",numpy.min(dws))
				if num_mb*batch_size > 30 or num_mb>int(X.shape[0]/batch_size):
					batch_done=True
				j+=1
				#print(j)
				if j>=m-1:
					j=0
				
				
	
			self.n_iter+=1
			self.t+=1	
			
			step=self.learning_rate*dw/numpy.sqrt((dws+1e-8)/(num_mb*batch_size))#/(batch_size*num_mb))
			x=x-step
			self.update_params_from_1darr(x)
			self.save.append(self.model.j)
			#print(verbose)
			if verbose:
				strr="Epoch "+str(i+1)+": " + str(int(100.0*float(j)/m))+ " %. Loss: "+str(score)
				if self.model.task==1:
					strr+=". Acc: "+str(calc_acc(batch_y,self.model.layers[-1].out))
				else:
					strr+=". r2: " + str(r2(batch_y,self.model.layers[-1].out))
				print(strr+str(num_mb))
			if self.n_iter>=maxiter:
				brflag=True
				break
		return numpy.array(self.save)
	
class Adam(Optimizer):
	def __init__(self,learning_rate=1e-3,beta_1=0.9,beta_2=0.999,epsilon=1e-8):
		super()
		self.learning_rate=learning_rate
		self.beta_1=beta_1
		self.beta_2=beta_2
		self.epsilon=1e-8
		self.first_run=True
	def reset(self):
		self.init_moments()
	def init_moments(self):	
		self.m1=numpy.zeros_like(self.weights_as_arr())
		self.m2=numpy.zeros_like(self.weights_as_arr())		
		self.t=0
		
	def fit(self,X,Y,model,batch_size=16,maxiter=100,verbose=True):
		self.model=model
		self.n_iter=0
		if self.first_run:
			self.init_moments()
			self.first_run=False
		brflag=False
		self.save=[]
		
		for i in range(0,100000):
			m=int(X.shape[0]/batch_size)
			for j in range(0,m):
				batch_x=X[j*batch_size:(j+1)*batch_size]
				batch_y=Y[j*batch_size:(j+1)*batch_size]
				
				#Calc raw gradients
				self.model.update(batch_x,batch_y)
				
				x=self.weights_as_arr()
				dw=self.weight_grads_as_arr()
				
				self.n_iter+=1
				
				#Adam
				self.t+=1
				
				self.m1=self.beta_1*self.m1+(1.0-self.beta_1)*dw
				self.m2=self.beta_2*self.m2+(1.0-self.beta_2)*(dw-self.m1)**2/numpy.abs(1.0/numpy.log(self.beta_1))
				
				m1a=self.m1/(1.0-self.beta_1**self.t)
				m2a=self.m2/(1.0-self.beta_2**self.t)
				
				x=x-self.learning_rate*self.m1/numpy.sqrt(self.m2+1e-8)
				self.update_params_from_1darr(x)
				self.save.append(self.model.j)
				if verbose:
					strr="Epoch "+str(i+1)+": " + str(int(100.0*float(j)/m))+ " %. Loss: "+str(self.model.j)
					print(strr)
				if self.n_iter>=maxiter:
					brflag=True
					break
			if brflag:
				break
		return numpy.array(self.save)
class SciPyMin(Optimizer):
	def __init__(self,method):
		super()
		self.method=method
	def objfn(self,x):
		self.update_params_from_1darr(x)

		self.preds=self.model.update(self.X,self.Y)
		
		return self.model.j,self.weight_grads_as_arr()
		
	def print_msg(self,x):
		if self.verbose:
			strr="Epoch " +str(self.epoch)+" . Cost: " + str(self.model.j)
			#if self.model.task==1:
			#	strr+=". Acc: "+str(calc_acc(self.Y,self.preds))
			#else:
			#	strr+=". r2: " + str(r2(self.Y,self.preds))
			print(strr)
	
		self.epoch+=1
			
	def fit(self,X,Y,model,batch_size=None, maxiter=100,verbose=True):
		self.X=X
		self.Y=Y
		self.model=model
		self.verbose=verbose
		self.epoch=1
			
		from scipy.optimize import minimize
		x0=self.weights_as_arr()
		res=minimize(self.objfn,x0,jac=True,method=self.method,tol=1e-16,options={'maxiter':maxiter},callback=self.print_msg)
		self.update_params_from_1darr(res['x'])

class SDProp(Optimizer):
	def __init__(self,learning_rate=1e-3,beta_1=0.9,beta_2=0.99,epsilon=1e-8,num_bands=5):
		super()
		self.learning_rate=learning_rate
		self.beta_1=beta_1
		self.beta_2=beta_2
		self.epsilon=1e-8
		self.first_run=True
		self.expb=False
		self.num_bands=num_bands
	def reset(self):
		self.init_moments()
	def init_moments(self):	
		self.m1=0.0
		self.m12=0.0
		self.m2=0.0
		self.covm=0.0

		self.save=[]
		self.t=0
			
	def fit(self,X,Y,model,batch_size=16,maxiter=100,verbose=True):
		self.model=model
		self.n_iter=0
		if self.first_run:
			self.init_moments()
			self.first_run=False
		brflag=False

		for i in range(0,100000):
			m=int(X.shape[0]/batch_size)
			for j in range(0,m):
				batch_x=X[j*batch_size:(j+1)*batch_size]
				batch_y=Y[j*batch_size:(j+1)*batch_size]
				#Calc raw gradients
				self.model.update(batch_x,batch_y)
				
				x=self.weights_as_arr()
				dw=self.weight_grads_as_arr()
				
				
				#Calc raw gradients
				self.model.update(batch_x,batch_y)
				self.t+=1
				self.n_iter+=1
				
				self.m1=self.beta_1*self.m1+(1.0-self.beta_1)*dw
				self.m12=self.beta_2*self.m2+(1.0-self.beta_2)*dw
				self.m2=self.beta_2*self.m2+(1.0-self.beta_2)*dw**2
				
				
				m1a=self.m1#/(1.0-self.beta_1**self.t)
				m12a=self.m12#/(1.0-self.beta_2**self.t)
				m2a=self.m2#/(1.0-self.beta_2**self.t)
				#/(1.0-self.beta_2**self.t)
				
				if self.expb:

					dwt=dw
					m12t=self.m12
					self.covm=self.beta_2*self.covm+(1.0-self.beta_2)*numpy.outer(dwt-m12t,dwt-m12t)

					w,v=eigh(self.covm+numpy.identity(len(self.covm))*1e-8)

					s2=w**(-0.5)
					s2=numpy.diag(s2)
					K=numpy.dot(v,numpy.dot(s2,v.T))
					step=self.learning_rate*numpy.dot(K,dwt)
					x=x-step
				
				else:
				
					x=x-self.learning_rate*dw/(numpy.sqrt(m2a)+1e-8)
				self.update_params_from_1darr(x)
				
				self.save.append(self.model.j)
				if verbose:
					strr="Epoch "+str(i+1)+": " + str(int(100.0*float(j)/m))+ " %. Loss: "+str(self.model.j)

					print(strr)
				if self.n_iter>=maxiter:
					brflag=True
					break
			if brflag:
				break
		return numpy.array(self.save)

from sklearn.cluster import KMeans
class Adam2(Optimizer):
	def __init__(self,learning_rate=1e-3,beta_1=0.9,beta_2=0.999,epsilon=1e-8):
		super()
		self.learning_rate=learning_rate
		self.beta_1=beta_1
		self.beta_2=beta_2
		self.epsilon=1e-8
		self.first_run=True
	def reset(self):
		self.init_moments()
	def init_moments(self):	
		self.m1=numpy.zeros_like(self.weights_as_arr())
		self.m2=numpy.zeros_like(self.weights_as_arr())		
		self.t=0
	def fit(self,X,Y,model,batch_size=16,maxiter=100,verbose=True):
		self.model=model
		self.n_iter=0
		if self.first_run:
			self.init_moments()
			self.first_run=False
		brflag=False
		self.save=[]
		
		

		m=int(X.shape[0]/batch_size)
		for i in range(0,100000):
			A=self.model.fast_forward(X)
			
			jj=numpy.random.randint(A.shape[0])
			
			d=numpy.sum((A[[jj]]-A)**2,1)
			ass=numpy.argsort(d)
			
			batch_x=X[ass[0:500]]
			batch_y=Y[ass[0:500]]
			
			
			self.model.update(batch_x,batch_y)
			
			x=self.weights_as_arr()
			dw=self.weight_grads_as_arr()
			
			self.n_iter+=1
			
			#Adam
			self.t+=1
			
			self.m1=self.beta_1*self.m1+(1.0-self.beta_1)*dw
			self.m2=self.beta_2*self.m2+(1.0-self.beta_2)*(dw-self.m1)**2/numpy.abs(1.0/numpy.log(self.beta_1))
			
			m1a=self.m1/(1.0-self.beta_1**self.t)
			m2a=self.m2/(1.0-self.beta_2**self.t)
			
			x=x-self.learning_rate*self.m1/numpy.sqrt(self.m2+1e-8)
			self.update_params_from_1darr(x)
			self.save.append(self.model.j)
			if verbose:
				strr= str(i)+" "+str(int(100.0*float(i)/m))+ " %. Loss: "+str(self.model.j) + " " +str(jj)
				print(strr)
			if self.n_iter>=maxiter:
				brflag=True
				break
		return numpy.array(self.save)
