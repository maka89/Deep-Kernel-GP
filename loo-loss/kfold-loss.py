import torch
from torch.autograd import Variable,gradcheck
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from utils import load_mnist
from scipy.sparse.linalg import bicgstab,bicg
from scipy.sparse import csr_matrix,vstack,lil_matrix
import time
np.random.seed(0)
n_folds=6

(xnp,ynp),(x_test,y_test) = load_mnist()
xnp=xnp.reshape(-1,28*28).astype(np.float64)

ynp=ynp.astype(np.float64)
ynp=np.argmax(ynp,1).reshape(-1,1).astype(np.float64)
x_test=x_test.reshape(-1,28*28).astype(np.float64)
y_test=np.argmax(y_test,1).reshape(-1,1).astype(np.float64)
y_test=y_test.astype(np.float64)


x_test=Variable(torch.from_numpy(x_test),requires_grad=False)
y_test=Variable(torch.from_numpy(y_test),requires_grad=False)

nn=5000
x,y,alphas=[],[],[]
for i in range(0,n_folds):
	x.append(Variable(torch.from_numpy(xnp[i*nn:(i+1)*nn]),requires_grad=False) )
	y.append(Variable(torch.from_numpy(ynp[i*nn:(i+1)*nn]),requires_grad=False) )
	alphas.append(np.zeros((nn*(n_folds-1),ynp.shape[1])))

#Learnable parameters
lamb = 3.5e-1


hidden_size=64
n_out=10
w1,b1 = Variable(torch.from_numpy(np.random.normal(0.0,1.0,size=(xnp.shape[1],hidden_size))/np.sqrt(xnp.shape[1])),requires_grad=True),Variable(torch.from_numpy(np.zeros((1,hidden_size))),requires_grad=True)
w2,b2 = Variable(torch.from_numpy(np.random.normal(0.0,1.0,size=(hidden_size,hidden_size))/np.sqrt(hidden_size)),requires_grad=True),Variable(torch.from_numpy(np.zeros((1,hidden_size))),requires_grad=True)
w3,b3 = Variable(torch.from_numpy(np.random.normal(0.0,1.0,size=(hidden_size,n_out))/np.sqrt(hidden_size)),requires_grad=True),Variable(torch.from_numpy(np.zeros((1,n_out))),requires_grad=True)

def neural_net(x):
	h=torch.tanh(torch.mm(x,w1)+b1)
	h=torch.tanh(torch.mm(h,w2)+b2)
	z=torch.mm(h,w3)+b3	
	return z

def pred(z_test,z_train,alpha):
	ztestnp=z_test.data.numpy()
	ztrainnp=z_train.data.numpy()
	pred=np.zeros(z_test.shape[0])
	for i in range(0,ztestnp.shape[0]):
		tmp=np.exp(-0.5*np.sum((ztestnp[i]-ztrainnp)**2,1) ) 
		pred[i] = np.dot(tmp.reshape(1,-1),alpha.reshape(-1,1))
	print(pred.shape)
	return pred
def kernel_np(z):
	tmp=[]
	for i in range(0,z.shape[1]):
		dists=z[:,[i]].T-z[:,[i]]
		tmp.append(dists.reshape(dists.shape[0],dists.shape[1],1))
	dists=np.concatenate(tmp,2)
	return np.exp(-0.5*np.sum(dists**2,2))

def kernel_sparse(z):
	K=np.zeros((z.shape[0],z.shape[0]))
	for i in range(0,z.shape[0]):
		K[i,:]=np.exp(-0.5*np.sum((z[i]-z)**2,1))
		K[i,i]+=lamb
	return K


def kernel(z):
	tmp=[]
	for i in range(0,z.size()[1]):
		dists=torch.t(z[:,[i]])-z[:,[i]]
		tmp.append(dists.view(dists.size()[0],dists.size()[1],1))
	dists=torch.cat(tmp,2)
	return torch.exp(-0.5*torch.sum(dists**2,2))

def kernel2(z1,z2):
	tmp=[]
	for i in range(0,z1.size()[1]):
		dists=z1[:,[i]]-torch.t(z2[:,[i]])
		tmp.append(dists.view(dists.size()[0],dists.size()[1],1))
	dists=torch.cat(tmp,2)
	return torch.exp(-0.5*torch.sum(dists**2,2))


t1=0.0
t2=0.0
def loss_fn(x,y):
	global t1,t2
	la2=[]
	for i in range(0,len(x)):

		x_val,y_val = x[i],y[i]
		
		idx=np.delete(np.arange(len(x)),i)
		x_train,y_train = x[idx[0]],y[idx[0]]
		for j in range(1,len(idx)):
			x_train=torch.cat([x_train,x[j]],0)
			y_train=torch.cat([y_train,y[j]],0)

		z_train=neural_net(x_train).detach()

		z_val=neural_net(x_val)

		tic=time.time()
		Knp=kernel_sparse(z_train.data.numpy())
		t1+=time.time()-tic
		#Knp=kernel_np(z_train.data.numpy())+np.identity(z_train.size()[0])*lamb
		#Knp=(Knp>=1e-3)*Knp
		#Knp=csr_matrix(Knp)
		tic=time.time()
		la=[]
		for j in range(0,y_train.size()[1]):
			#print(i,j,y_train.shape,alphas[i][:,j].shape,Knp.shape)
			alphas[i][:,j],info=bicgstab(Knp,y_train[:,j].data.numpy(),x0=alphas[i][:,j],tol=1e-6)
			
			K2 = kernel2(z_val,z_train)
			y_pred= torch.mm(K2, Variable(torch.from_numpy(alphas[i][:,j].reshape(-1,1)),requires_grad=False).detach())
			
			loss = torch.mean((y_val-y_pred)**2)

			la.append(loss)
		fl=torch.mean(torch.cat(la))
		la2.append(fl)
		t2+=time.time()-tic
	final_loss=torch.mean(torch.cat(la2))
	
	return final_loss




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

for t in range(0,10000):



	loss=loss_fn(x,y)
	print(t,loss.data.numpy(),lamb,t1,t2)
	if t >0:
		#lamb.grad.data.zero_()
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
	#dl2=beta_2*dl2 + (1.0-beta_2)*lamb.grad.data**2

	dw11=beta_1*dw11 + (1.0-beta_1)*w1.grad.data
	db11=beta_1*db11 + (1.0-beta_1)*b1.grad.data
	dw21=beta_1*dw21 + (1.0-beta_1)*w2.grad.data
	db21=beta_1*db21 + (1.0-beta_1)*b2.grad.data
	dw31=beta_1*dw31 + (1.0-beta_1)*w3.grad.data
	db31=beta_1*db31 + (1.0-beta_1)*b3.grad.data
	#dl1=beta_1*dl1 + (1.0-beta_1)*lamb.grad.data

	#lamb.data = lamb.data -learning_rate*dl1/(np.sqrt(dl2)+1e-8)
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
	if t % 10==0:
		final_pred=np.zeros_like(y_test.data.numpy())
		la2=[]
		for l in range(0,len(x)):

			x_val,y_val = x[l],y[l]
		
			idx=np.delete(np.arange(len(x)),l)
			x_train,y_train = x[idx[0]],y[idx[0]]
			for j in range(1,len(idx)):
				x_train=torch.cat([x_train,x[j]],0)
				y_train=torch.cat([y_train,y[j]],0)

			z_train=neural_net(x_train).detach()
			z_test=neural_net(x_test)

			la=[]
			for j in range(0,y_train.size()[1]):
				y_pred= pred(z_test,z_train,alphas[l][:,j])
				loss = np.mean((y_test.data.numpy()-y_pred)**2)
				final_pred[:,j]+=y_pred

				la.append(loss)

			fl=np.mean(np.array(la))
			la2.append(fl)
		final_pred/=n_folds
		final_loss=np.mean(np.array(la2))
		print(np.sqrt(final_loss))
		print(np.average(np.rint(final_pred)==y_test.data.numpy()))
	
