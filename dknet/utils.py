import numpy

def one_hot(x,n_classes):
	assert(len(x.shape)==1)
	A=numpy.zeros((x.shape[0],n_classes))
	A[numpy.arange(len(x)),x]=1.0
	return A
def calc_acc(y_true,y_pred):
	if y_true.shape[1] > 1:
		return numpy.average(numpy.argmax(y_true,1)==numpy.argmax(y_pred,1))
	else:
		return numpy.average(1.0*(y_pred>=0.5) == y_true)
def r2(y_true,y_pred):
	avg = numpy.mean(y_true,0)
	var = numpy.sum((y_true-avg)**2,0)
	err = numpy.sum((y_true-y_pred)**2,0)
	r2=1.0-err/var
	#print(r2)
	return r2
def normalize(X,sub,div):
		return (numpy.copy(X)-sub)/div

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
	
def load_cifar(shuffle=False):
	x_train=numpy.zeros((0,32,32,3))
	y_train=numpy.zeros((0,),dtype=numpy.int)
	x_test=numpy.zeros((0,32,32,3))
	y_test=numpy.zeros((0,),dtype=numpy.int)
	for i in range(0,5):
		dat=unpickle("data/cifar10/data_batch_"+str(i+1))
		print("KEYS: ")
		print(dat.keys())
		xdat=numpy.zeros((len(dat[b'data']),32,32,3))
		xdat[:,:,:,0]=dat[b'data'][:,0:1024].reshape(-1,32,32)
		xdat[:,:,:,1]=dat[b'data'][:,1024:2048].reshape(-1,32,32)
		xdat[:,:,:,2]=dat[b'data'][:,2048:3072].reshape(-1,32,32)
		x_train=numpy.concatenate((x_train,xdat),0)
		y_train=numpy.concatenate((y_train,dat[b"labels"]))
		
	dat=unpickle("data/cifar10/test_batch")
	xdat=numpy.zeros((len(dat[b'data']),32,32,3))
	xdat[:,:,:,0]=dat[b'data'][:,0:1024].reshape(-1,32,32)
	xdat[:,:,:,1]=dat[b'data'][:,1024:2048].reshape(-1,32,32)
	xdat[:,:,:,2]=dat[b'data'][:,2048:3072].reshape(-1,32,32)
	x_test=numpy.concatenate((x_test,xdat),0)
	y_test=numpy.concatenate((y_test,dat[b"labels"]))
	
	x_train=x_train.astype('float32')
	x_test=x_test.astype('float32')
	x_train /= 255.0
	x_test /= 255.0
	
	y_train=y_train.astype('int')
	y_test=y_test.astype('int')
	print(y_train)
	y_train = one_hot(y_train, 10)
	y_test = one_hot(y_test, 10)
	
	if shuffle:
		#Shuffle data. 
		tmp=numpy.arange(len(x_train))
		numpy.random.shuffle(tmp)
		x_train,y_train=x_train[tmp],y_train[tmp]
		
		tmp=numpy.arange(len(x_test))
		numpy.random.shuffle(tmp)
		x_test,y_test=x_test[tmp],y_test[tmp]
		
	return [[x_train,y_train],[x_test,y_test]]
def load_mnist(shuffle=False):
	
	#If error loading files, use this to aquire mnist, if you have keras.
	#
	#from keras.datasets import mnist
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#numpy.savez_compressed("data/mnist/mnist_train",a=x_train,b=y_train)
	#numpy.savez_compressed("data/mnist/mnist_test",a=x_test,b=y_test)

	tftr,tfte=numpy.load("data/mnist/mnist_train.npz"),numpy.load("data/mnist/mnist_test.npz")
	x_train,y_train=tftr['a'],tftr['b']
	x_test,y_test=tfte['a'],tfte['b']
	
	x_train=x_train.astype('float32').reshape(-1,28,28,1)
	x_test=x_test.astype('float32').reshape(-1,28,28,1)
	x_train /= 255.0
	x_test /= 255.0
	y_train = one_hot(y_train, 10)
	y_test = one_hot(y_test, 10)
	
	if shuffle:
		#Shuffle data. 
		tmp=numpy.arange(len(x_train))
		numpy.random.shuffle(tmp)
		x_train,y_train=x_train[tmp],y_train[tmp]
		
		tmp=numpy.arange(len(x_test))
		numpy.random.shuffle(tmp)
		x_test,y_test=x_test[tmp],y_test[tmp]
	
	return [[x_train,y_train],[x_test,y_test]]
	
	
def grad_check(model,X,Y,check_n_params=50):
	eps=1e-7
	
	ll=[]
	for n in range(0,check_n_params):
		model.forward(X,gc=True)
		model.backward(Y)
		i=numpy.random.randint(len(model.layers))
		while not model.layers[i].trainable:
			i=numpy.random.randint(len(model.layers))
		nums=[]
		for j in range(0,len(model.layers[i].W.shape)):
			nums.append(numpy.random.randint(model.layers[i].W.shape[j]))
		nums=tuple(nums)
		
		bnum=[]
		for j in range(0,len(model.layers[i].b.shape)):
			bnum.append(numpy.random.randint(model.layers[i].b.shape[j]))
		bnum=tuple(bnum)
		
		dW=model.layers[i].dW.item(nums)
		db=model.layers[i].db.item(bnum)
		W=numpy.copy(model.layers[i].W)
		b=numpy.copy(model.layers[i].b)
		
		model.layers[i].W.itemset(nums,W.item(nums)+eps)
		model.forward(X,gc=True)
		model.backward(Y)
		jp=model.j
		
		model.layers[i].W.itemset(nums,W.item(nums)-eps)
		model.forward(X,gc=True)
		model.backward(Y)
		jm=model.j
		model.layers[i].W.itemset(nums,W.item(nums))
		
		dW2=0.5*(jp-jm)/eps
		
		model.layers[i].b.itemset(bnum,b.item(bnum)+eps)
		model.forward(X,gc=True)
		model.backward(Y)
		jp=model.j
		model.layers[i].b.itemset(bnum,b.item(bnum)-eps)
		model.forward(X,gc=True)
		model.backward(Y)
		jm=model.j
		
		db2=0.5*(jp-jm)/eps
		model.layers[i].b.itemset(bnum,b.item(bnum))
		tmp=[numpy.abs(db2-db),numpy.abs(dW2-dW)]
		ll.append(tmp)
	#print(ll)
	ll=numpy.array(ll)
	return numpy.max(ll,0)