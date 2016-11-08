import numpy as np
from copy import deepcopy
import random, math, sys
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn import svm

numoffeatures = 10000
newnumoffeatures = 100
numofdata = 100
numofvaliddata = 100

# numoffeatures = 500
# newnumoffeatures = 100
# numofdata = 2000
# numofvaliddata = 600

def populatedata(values, filename, numofdatas):
	fp = open(filename)
	data = fp.readlines()

	for i in range(numofdatas):
		line = data[i].split(' ')
		line = line[:len(line) - 1]
		if line == ['']:
			continue
		line = [int(x) for x in line]
		
		for j in range(len(line)):
			values[i][j] = line[j]

def getlabels(filename, ranges):
	fp = open(filename)
	data = fp.readlines()

	labels = []
	for i in range(ranges):
		label = int(data[i])
		labels.append(label)

	return labels

def kernelpca(data, gamma):
	squaredistances = pdist(data, 'sqeuclidean')

	sqdistmatrix = squareform(squaredistances)

	kernel = exp(-gamma * sqdistmatrix)

	onen = np.ones((numofdata, numofdata)) / numofdata
	kernel = kernel - onen.dot(kernel) - kernel.dot(onen) + onen.dot(kernel).dot(onen)

	eigvals, eigvecs = eigh(kernel)

	topeigvecs = np.column_stack((eigvecs[:,-i] for i in range(1,newnumoffeatures+1)))
	topeigvals = [eigvals[-i] for i in range(1,newnumoffeatures+1)]

	projdata = np.column_stack((eigvecs[:,-i] for i in range(1, newnumoffeatures+1)))
	return topeigvecs, topeigvals, kernel, projdata

def projvaliddata(X, alpha, eigvals, gamma, data, labels):
	validproj = []
	for i in data:
		dist = np.array([np.sum((i - row)**2) for row in X])
		k = np.exp(-gamma * dist)
		validproj.append(k.dot(alpha / eigvals))

	return np.array(validproj)

if __name__ == '__main__':
	random.seed()

	data = np.zeros((numofdata, numoffeatures))
	populatedata(data, 'arcene_train.data', numofdata)
	# populatedata(data, 'madelon_train.data', numofdata)

	labels = getlabels('arcene_train.labels', numofdata)
	# labels = getlabels('madelon_train.labels', numofdata)

	eigvecs, eigvals, kernel, projdata = kernelpca(data, 0.00001)


	validdata = np.zeros((numofvaliddata, numoffeatures))
	populatedata(validdata, 'arcene_valid.data', numofvaliddata)
	# populatedata(validdata, 'madelon_valid.data', numofvaliddata)

	validlabels = getlabels('arcene_valid.labels', numofvaliddata)
	# validlabels = getlabels('madelon_valid.labels', numofvaliddata)

	projvaliddata = projvaliddata(data, eigvecs, eigvals, 0.00001, validdata, validlabels)

	C = 1.0
	clf = svm.SVC(kernel='rbf', gamma=0.00001, C=C)
	svc = clf.fit(projdata, labels)

	results = clf.predict(projvaliddata)

	count = 0
	for i in range(len(results)):
		if results[i] == validlabels[i]:
			count += 1

	print (float(count)/numofvaliddata)*100,"%"

	# from sklearn.decomposition import PCA, KernelPCA
	# kpca = KernelPCA(kernel="poly", degree=2, n_components=10)
	# Train_KPCA1 = kpca.fit_transform(data)
	# Test1 = kpca.transform(validdata)

	# C = 1.0
	# clf = svm.SVC(kernel='rbf', gamma=0.00001, C=C)
	# svc = clf.fit(Train_KPCA1, labels)

	# results = clf.predict(Test1)

	# count = 0
	# for i in range(len(results)):
	# 	if results[i] == validlabels[i]:
	# 		count += 1

	# print (float(count)/numofvaliddata)*100,"%"


	# plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == 1], [newdata[i][1] for i in range(len(newdata)) if labels[i] == 1], color='red', alpha=0.5)
	# plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == -1], [newdata[i][1] for i in range(len(newdata)) if labels[i] == -1], color='blue', alpha=0.5)
	# plt.show()
