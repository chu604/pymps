# -*- coding: utf-8 -*-
# @Author: guochu
# @Date:   2017-11-21 09:41:34
# @Last Modified by:   1000787
# @Last Modified time: 2018-05-01 16:23:25
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.insert(0, lib_path)
from mpslearn.classifier import mpsclassifier
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from numpy import array
from scipy.linalg import norm
import h5py
from mnist import MNIST
import pickle

mndata = MNIST('./data')
xtrain, ytrain = mndata.load_training()
xtest, ytest = mndata.load_testing()

# xtrain = xtrain[:200]
# ytrain = ytrain[:200]

# xtest = xtest[:10]
# ytest = ytest[:10]

# renormalize each image
# for image, the overall norm does not matter
for i in range(len(xtrain)):
	xtrain[i] = xtrain[i]/norm(xtrain[i])

for i in range(len(xtest)):
	xtest[i] = xtest[i]/norm(xtest[i])

# with open('mnist_xtrain.pickle', 'wb') as f:
# 	pickle.dump(array(xtrain), f, pickle.HIGHEST_PROTOCOL)

# with open('mnist_ytrain.pickle', 'wb') as f:
# 	pickle.dump(array(ytrain), f, pickle.HIGHEST_PROTOCOL)

# with open('mnist_xtest.pickle', 'wb') as f:
# 	pickle.dump(array(xtest), f, pickle.HIGHEST_PROTOCOL)

# with open('mnist_ytest.pickle', 'wb') as f:
# 	pickle.dump(array(ytest), f, pickle.HIGHEST_PROTOCOL)


# bond dimension
D = 10
alpha = 0.001
# maximum number of iterations
kmax = 10
# tolerance of convergence
tol=1.0e-5

clf = mpsclassifier(D=D)
clf.train(xtrain, ytrain, less_memory=False, alpha=alpha, kmax=kmax, tol=tol, verbose=2)

temp = array(clf.predict(xtrain))
scores_train = (temp == ytrain)
train_success_rate = sum(scores_train)/len(scores_train)

temp = array(clf.predict(xtest))
scores_test = (temp == ytest)
test_success_rate = sum(scores_test)/len(scores_test)

print('the testing scores are:', scores_test)
print('test success rate', test_success_rate)

save_name = 'results/mpslearn_mnistD' + str(D) + 'kmax' + str(kmax) + 'alpha' + str(alpha) + '.h5' 
file = h5py.File(save_name, 'w')
file.create_dataset(name="scores_train", data=scores_train)
file.create_dataset(name="scores_test", data=scores_test)
file.create_dataset(name="train_success_rate", data=train_success_rate)
file.create_dataset(name="test_success_rate", data=test_success_rate)
file.create_dataset(name="iterations", data=clf.iterations)
file.create_dataset(name="error", data=clf.error)
file.create_dataset(name="kvals", data=clf.kvals)


