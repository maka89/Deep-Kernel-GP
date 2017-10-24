import numpy as np
import matplotlib.pyplot as plt
from dknet import NNRegressor
from dknet.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,CovMat
from dknet.optimizers import Adam,SciPyMin,SDProp
from dknet.utils import load_mnist

from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel

(x_train,y_train),(x_test,y_test)=load_mnist(shuffle=True)
x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)



layers=[]
layers.append(Dense(64,activation='relu'))
layers.append(Dense(64,activation='relu'))

layers.append(Dense(10))
layers.append(CovMat(alpha=0.5,var=1.0,kernel='rbf'))
opt=Adam(1e-3)
n_train = 5000
n_test = 10000
gp=NNRegressor(layers,opt=opt,batch_size=1000,maxiter=100,gp=True,verbose=True)
gp.fit(x_train,y_train)

gp.fit(x_train,y_train,batch_size=5000,maxiter=5)


#Can extract mapping z(x) and hyperparams for use in other learning algorithm
alph=gp.layers[-1].s_alpha
var=gp.layers[-1].var
A=gp.fast_forward(x_train[0:n_train])

kernel=ConstantKernel(var)*RBF(np.ones(1))+WhiteKernel(alph)
gp1=GaussianProcessRegressor(kernel,optimizer=None)
gp1.fit(A,y_train[0:n_train])


A=gp.fast_forward(x_test[0:n_test])
yp,std=gp1.predict(A,return_std=True)
print("GP Regression:")
print(np.average(np.argmax(yp,1)==np.argmax(y_test[0:n_test],1)))




ass=np.argsort(std)
f,ax=plt.subplots(3)
ax[0].plot(std[ass])
arr=1.0*(np.argmax(yp,1)==np.argmax(y_test[0:n_test],1))[ass]
ax[1].plot(np.argwhere(arr>=0.5)[:,0],arr[arr>=0.5],'b.')
ax[1].plot(np.argwhere(arr<0.5)[:,0],arr[arr<0.5],'r.')
ax[2].plot(np.cumsum(arr)/(np.arange(len(arr))+1))

ax[0].set_ylabel("Estimated\nstd.dev")
ax[1].set_ylabel("Correct\nprediction\n(bool)")
ax[2].set_ylabel("Cumulative\naccuracy")
f.suptitle("Test data sorted by estimated std.dev.")
plt.show(block=True)