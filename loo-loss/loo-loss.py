import torch
from torch.autograd import Variable,gradcheck
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from utils import load_mnist

class KernelInverse(torch.autograd.Function):

	def forward(self,input):
		
		K=input.numpy()
		L=cholesky(K,lower=True)
		L_inv=solve_triangular(L.T,np.eye(L.shape[0]))
		K_inv=L_inv.dot(L_inv.T)
		
		Kinv=torch.from_numpy(K_inv)
		self.save_for_backward(Kinv)
		return Kinv
	def backward(self,grad_output):
		Kinv, = self.saved_tensors
		return -torch.mm(Kinv,torch.mm(grad_output,Kinv))

np.random.seed(0)



#xnp=np.random.random((15,1))-0.5
#ynp=np.sin(10.0*xnp)
#ynp=ynp+np.random.normal(0.0,0.15,size=ynp.shape)


(xnp,ynp),(x_test,y_test) = load_mnist()
xnp=xnp.reshape(-1,28*28).astype(np.float64)

ynp=ynp.astype(np.float64)
ynp=np.argmax(ynp,1).reshape(-1,1).astype(np.float64)
x_test=x_test.reshape(-1,28*28).astype(np.float64)
y_test=np.argmax(y_test,1).reshape(-1,1).astype(np.float64)
y_test=y_test.astype(np.float64)

#Learnable parameters
lamb = Variable(torch.from_numpy(np.ones((1,1)))*-0.5, requires_grad=True)
hidden_size=64
n_out=50
w1,b1 = Variable(torch.from_numpy(np.random.normal(0.0,1.0,size=(xnp.shape[1],hidden_size))/np.sqrt(xnp.shape[1])),requires_grad=True),Variable(torch.from_numpy(np.zeros((1,hidden_size))),requires_grad=True)
w2,b2 = Variable(torch.from_numpy(np.random.normal(0.0,1.0,size=(hidden_size,hidden_size))/np.sqrt(hidden_size)),requires_grad=True),Variable(torch.from_numpy(np.zeros((1,hidden_size))),requires_grad=True)


w3,b3 = Variable(torch.from_numpy(np.random.normal(0.0,1.0,size=(hidden_size,n_out))/np.sqrt(hidden_size)),requires_grad=True),Variable(torch.from_numpy(np.zeros((1,n_out))),requires_grad=True)




def kernel(x):
	h=torch.tanh(torch.mm(x,w1)+b1)
	h=torch.tanh(torch.mm(h,w2)+b2)
	z=torch.mm(h,w3)+b3	
	#z=10.0*z
	#z=x
	tmp=[]
	for i in range(0,z.size()[1]):
		dists=torch.t(z[:,[i]])-z[:,[i]]
		tmp.append(dists.view(dists.size()[0],dists.size()[1],1))
	dists=torch.cat(tmp,2)
	return torch.exp(-0.5*torch.sum(dists**2,2))#torch.mm(torch.t(x),x)

def hat_matrix(x,lamb):
	K=kernel(x)
	K2=K+Variable(torch.from_numpy(np.eye(K.size()[0])),requires_grad=True)*(1.0/(torch.exp(-lamb[0])+1.0)+1e-8)
	Kinv=KernelInverse()(K2)
	hat=torch.mm(K,Kinv)
	return hat

def loss_fn(hat,y):
	la=[]
	for i in range(0,y.size()[1]):
		matt= Variable(torch.from_numpy(np.eye(hat.size()[0])),requires_grad=True) - hat
		fac1=torch.mm(y[:,i].contiguous().view(1,y.size()[0]),matt)
		fac2=torch.mm(matt,y[:,i].contiguous().view(y.size()[0],1))

		diagg = torch.diag(1.0/torch.diag(matt))
		fac1=torch.mm(fac1,diagg)
		fac2=torch.mm(diagg,fac2)

		loss=torch.mm(fac1,fac2)/y.size()[0]
		la.append(loss)
	fl=torch.mean(torch.cat(la))
	return fl


def f(xx):
	#xx=Variable(torch.from_numpy(np.random.random((5,1))-0.5), requires_grad=True)
	#y=Variable(torch.from_numpy(3.14*x.data.numpy()), requires_grad=False)

	#print(lamb)
	#lamb = Variable(torch.from_numpy(np.ones(1)*1e-2), requires_grad=False)
	#print(y.size())
	return loss_fn(hat_matrix(xx,lamb),y)




learning_rate=1e-3

dw12,db12=0.0,0.0
dw22,db22=0.0,0.0
dw32,db32=0.0,0.0
dl2 = 0.0

dw11,db11=0.0,0.0
dw21,db21=0.0,0.0
dw31,db31=0.0,0.0
dl1 = 0.0

beta_1=0.0
beta_2=0.99
batch_size=500
m=int(xnp.shape[0]/batch_size)
for t in range(0,10000):
	

	for j in range(0,m):
		
		x=Variable(torch.from_numpy(xnp[j*batch_size:(j+1)*batch_size]), requires_grad=True)
		y=Variable(torch.from_numpy(ynp[j*batch_size:(j+1)*batch_size]), requires_grad=False)
		#print(gradcheck(f,[x]))
		#print(x.size())
	
		loss=loss_fn(hat_matrix(x,lamb),y)
		print(t,loss.data.numpy(),1.0/(np.exp(-lamb.data.numpy()) + 1.0))
		if t >0:
			lamb.grad.data.zero_()
			w1.grad.data.zero_()
			w2.grad.data.zero_()
			w3.grad.data.zero_()
			b1.grad.data.zero_()
			b2.grad.data.zero_()
			b3.grad.data.zero_()
		loss.backward()
	

		dw12=beta_2*dw12 + (1.0-beta_2)*w1.grad.data**2
		db12=beta_2*db12 + (1.0-beta_2)*b1.grad.data**2
		dw22=beta_2*dw22 + (1.0-beta_2)*w2.grad.data**2
		db22=beta_2*db22 + (1.0-beta_2)*b2.grad.data**2
		dw32=beta_2*dw32 + (1.0-beta_2)*w3.grad.data**2
		db32=beta_2*db32 + (1.0-beta_2)*b3.grad.data**2
		dl2=beta_2*dl2 + (1.0-beta_2)*lamb.grad.data**2

		dw11=beta_1*dw11 + (1.0-beta_1)*w1.grad.data
		db11=beta_1*db11 + (1.0-beta_1)*b1.grad.data
		dw21=beta_1*dw21 + (1.0-beta_1)*w2.grad.data
		db21=beta_1*db21 + (1.0-beta_1)*b2.grad.data
		dw31=beta_1*dw31 + (1.0-beta_1)*w3.grad.data
		db31=beta_1*db31 + (1.0-beta_1)*b3.grad.data
		dl1=beta_1*dl1 + (1.0-beta_1)*lamb.grad.data

		lamb.data = lamb.data -learning_rate*dl1/(np.sqrt(dl2)+1e-8)
		w1.data = w1.data - learning_rate*dw11/(np.sqrt(dw12)+1e-8)
		b1.data = b1.data - learning_rate*db11/(np.sqrt(db12)+1e-8)
		w2.data = w2.data - learning_rate*dw21/(np.sqrt(dw22)+1e-8)
		b2.data = b2.data - learning_rate*db21/(np.sqrt(db22)+1e-8)
		w3.data = w3.data - learning_rate*dw31/(np.sqrt(dw32)+1e-8)
		b3.data = b3.data - learning_rate*db31/(np.sqrt(db32)+1e-8)

		#lamb.data = lamb.data -learning_rate*lamb.grad.data
		#w1.data = w1.data - learning_rate*w1.grad.data
		#b1.data = b1.data - learning_rate*b1.grad.data
		#w2.data = w2.data - learning_rate*w2.grad.data
		#b2.data = b2.data - learning_rate*b2.grad.data
		##w3.data = w3.data - learning_rate*w3.grad.data
		#b3.data = b3.data - learning_rate*b3.grad.data

		if j % 10 == 0:
			nug=lamb.data.numpy()
			K1=np.zeros((x_test.shape[0],xnp[j*batch_size:(j+1)*batch_size].shape[0]))
			K2=np.zeros((xnp[j*batch_size:(j+1)*batch_size].shape[0],xnp[j*batch_size:(j+1)*batch_size].shape[0]))

			

			z_test=np.tanh(np.dot(x_test,w1.data.numpy())+b1.data.numpy())
			z_test=np.tanh(np.dot(z_test,w2.data.numpy())+b2.data.numpy())
			z_test=np.dot(z_test,w3.data.numpy())+b3.data.numpy()
			#z_test=10.0*z_test

			znp=np.tanh(np.dot(xnp[j*batch_size:(j+1)*batch_size],w1.data.numpy())+b1.data.numpy())
			znp=np.tanh(np.dot(znp,w2.data.numpy())+b2.data.numpy())
			znp=np.dot(znp,w3.data.numpy())+b3.data.numpy()
			#znp=10.0*znp

			for k in range(0,xnp[j*batch_size:(j+1)*batch_size].shape[0]):
				K1[:,k]=np.exp(-0.5*np.sum((z_test-znp[[k],:])**2,1))
				K2[:,k]=np.exp(-0.5*np.sum((znp-znp[[k],:])**2,1))
				K2[k,k]+=(1.0/(np.exp(-nug)+1.0)+1e-8)
			L=cholesky(K2,lower=True)
			L_inv=solve_triangular(L.T,np.eye(L.shape[0]))
			K_inv=L_inv.dot(L_inv.T)
			
			yp=np.dot(K1,np.dot(K_inv,ynp[j*batch_size:(j+1)*batch_size]))
			yp2=np.rint(yp)
			print(np.average(yp2==y_test), np.sqrt(np.mean((yp-y_test)**2)) )

			#print(np.average(np.argmax(yp,1)==np.argmax(y_test,1)))
			#print(yp)
			#print(y_test)

	
