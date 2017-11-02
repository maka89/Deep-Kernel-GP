import numpy as np
import matplotlib.pyplot as plt
from dknet import NNRegressor
from dknet.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,CovMat,Scale
from dknet.optimizers import Adam,SciPyMin,SDProp, Adam2
from dknet.utils import load_mnist

from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel

(x_train,y_train),(x_test,y_test)=load_mnist(shuffle=True)
x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)

y_test=np.argmax(y_test,1).reshape(-1,1)
y_train=np.argmax(y_train,1).reshape(-1,1)

layers=[]
layers.append(Dense(64,activation='tanh'))
layers.append(Dense(64,activation='tanh'))
layers.append(Dense(20))
layers.append(CovMat(alpha=0.3,var=1.0,kernel='rbf'))
opt=SciPyMin('l-bfgs-b')
n_train = 3000
n_test = 10000


opt=Adam(1e-3)
batch_size=500
gp=NNRegressor(layers,opt=opt,batch_size=batch_size,maxiter=500,gp=True,verbose=True)
gp.fit(x_train,y_train)



#Can extract mapping z(x) and hyperparams for use in other learning algorithm
alph=gp.layers[-1].s_alpha
var=gp.layers[-1].var

A_full=gp.fast_forward(x_train)


kernel=ConstantKernel(var)*RBF(np.ones(1))+WhiteKernel(alph)



A_test=gp.fast_forward(x_test[0:n_test])
yp=np.zeros(n_test)
std=np.zeros(n_test)
gp1=GaussianProcessRegressor(kernel,optimizer=None)
gp1.fit(A_full[0:500],y_train[0:500])
mu,stdt=gp1.predict(A_test,return_std=True)

print("GP Regression:")
print( np.sqrt( np.mean( (np.rint(mu)-y_test)**2 ) ) ) 

