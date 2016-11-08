import numpy as np
from copy import deepcopy
import random, math, sys
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn import svm

# numoffeatures = 10000
# newnumoffeatures = 1
# numofdata = 100
# numofvaliddata = 100

numoffeatures = 500
newnumoffeatures = 1
numofdata = 2000
numofvaliddata = 600

def populatedata(values, filename, num):
	fp = open(filename)
	data = fp.readlines()

	for i in range(num):
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
	class1 = 0
	class2 = 0
	for i in range(ranges):
		label = int(data[i])
		if label == 1:
			class1 += 1
		else:
			class2 += 1
		labels.append(label)

	return labels, class1, class2

def kernellda(data, gamma, class1, class2, labels):
	squaredistances = pdist(data, 'sqeuclidean')

	sqdistmatrix = squareform(squaredistances)

	kernel = exp(-gamma * sqdistmatrix)

	onen = np.ones((numofdata, numofdata)) / numofdata
	kernel = kernel - onen.dot(kernel) - kernel.dot(onen) + onen.dot(kernel).dot(onen)

	#within class N
	K1 = np.zeros((numofdata, class1))
	K2 = np.zeros((numofdata, class2))
	for i in range(numofdata):
		K1count = 0
		K2count = 0
		for j in range(numofdata):
			if labels[j] == 1:
				K1[i][K1count] = kernel[i][j]
				K1count += 1
			else:
				K2[i][K2count] = kernel[i][j]
				K2count += 1

	oneK1 = np.ones((class1, class1))/class1
	oneK2 = np.ones((class2, class2))/class2

	N = K1.dot(np.identity(class1) - oneK1).dot(K1.T) + K2.dot(np.identity(class2) - oneK2).dot(K2.T)
	
	N = N + (np.identity(numofdata) * 0)

	#between class M
	M1 = np.zeros((numofdata, 1))
	M2 = np.zeros((numofdata, 1))

	for i in range(numofdata):
		sum1 = 0
		sum2 = 0
		for j in range(class1):
			sum1 += K1[i][j]
		for j in range(class2):
			sum2 += K2[i][j]
		M1[i] = float(sum1)/class1
		M2[i] = float(sum2)/class2

	product = np.linalg.inv(N).dot(M2 - M1)

	return product, kernel

def projvaliddata(X, alpha, gamma, data, labels):
    validproj = []
    for i in data:
        dist = np.array([np.sum((i - row)**2) for row in X])
        k = np.exp(-gamma * dist)
        validproj.append(k)

    validpoints = []
    for i in validproj:
    	validpoints.append(i.dot(alpha))

    return np.array(validpoints)

if __name__ == '__main__':
	random.seed()

	data = np.zeros((numofdata, numoffeatures))
	# populatedata(data, 'arcene_train.data', numofdata)
	populatedata(data, 'madelon_train.data', numofdata)	

	#class1 = +1 and class2 = -1
	# labels, class1, class2 = getlabels('arcene_train.labels', numofdata)
	labels, class1, class2 = getlabels('madelon_train.labels', numofdata)

	projection, kernel = kernellda(data, 0.00001, class1, class2, labels)

	newdata = kernel.dot(projection)

	validdata = np.zeros((numofvaliddata, numoffeatures))
	# populatedata(validdata, 'arcene_valid.data', numofdata)
	populatedata(validdata, 'madelon_valid.data', numofvaliddata)

	# validlabels, class1, class2 = getlabels('arcene_valid.labels', numofvaliddata)
	validlabels, class1, class2 = getlabels('madelon_valid.labels', numofvaliddata)

	projvaliddata = projvaliddata(data, projection, 0.00001, validdata, validlabels)

	C = 1.0
	clf = svm.SVC(kernel='rbf', gamma=0.00001, C=C)
	svc = clf.fit(newdata, labels)

	results = clf.predict(projvaliddata)

	count = 0
	for i in range(len(results)):
		if results[i] == validlabels[i]:
			count += 1

	print (float(count)/numofvaliddata)*100,"%"

	# plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == 1], [1 for i in range(len(newdata)) if labels[i] == 1], color='red', alpha=0.5)
	# plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == -1], [1 for i in range(len(newdata)) if labels[i] == -1], color='blue', alpha=0.5)
	# plt.show()
