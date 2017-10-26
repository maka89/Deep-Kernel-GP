import numpy as np
import matplotlib.pyplot as plt

from dknet import NNRegressor
from dknet.layers import Dense,CovMat,Dropout,Parametrize
from dknet.optimizers import Adam,SciPyMin,SDProp
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel
def f(x):
	return (x+0.5>=0)*np.sin(64*(x+0.5))#-1.0*(x>0)+numpy.

np.random.seed(0)
x_train=np.random.random(size=(70,1))-0.5
y_train=f(x_train)+np.random.normal(0.0,0.01,size=x_train.shape)



layers=[]

layers.append(Dense(32,activation='tanh'))
layers.append(Dense(10))
layers.append(CovMat(kernel='rbf'))

opt=Adam(1e-3)
#opt=SciPyMin('l-bfgs-b')

gp=NNRegressor(layers,opt=opt,batch_size=x_train.shape[0],maxiter=100,gp=True,verbose=True)
gp.fit(x_train,y_train)
x_test=np.linspace(-0.7,0.7,1000).reshape(-1,1)



y_pred,std=gp.predict(x_test)


plt.plot(x_test,gp.layers[-2].out)
plt.xlabel('X')
plt.ylabel('Z')
plt.figure()

plt.plot(x_train,y_train,'.')
plt.plot(x_test,f(x_test)[:,0])
plt.plot(x_test,y_pred)
plt.xlabel('X')
plt.ylabel('Y')
plt.fill_between(x_test[:,0],y_pred[:,0]-std,y_pred[:,0]+std,alpha=0.5)

plt.legend(['Training samples', 'True function', 'Predicted function','Prediction stddev'])
plt.show()
