import numpy as np
from copy import deepcopy
import random, math, sys
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

numoffeatures = 10000
newnumoffeatures = 1
numofdata = 100

def populatedata(values):
	fp = open('arcene_train.data')
	data = fp.readlines()

	for i in range(numofdata):
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
			class2 += 2
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
				K1[i][K1count] = data[i][j]
				K1count += 1
			else:
				K2[i][K2count] = data[i][j]
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

if __name__ == '__main__':
	random.seed()

	data = np.zeros((numofdata, numoffeatures))
	populatedata(data)

	#class1 = +1 and class2 = -1
	labels, class1, class2 = getlabels('arcene_train.labels', numofdata)

	projection, kernel = kernellda(data, 0.2, class1, class2, labels)

	newdata = kernel.dot(projection)

	plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == 1], [1 for i in range(len(newdata)) if labels[i] == 1], color='red', alpha=0.5)
	plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == -1], [1 for i in range(len(newdata)) if labels[i] == -1], color='blue', alpha=0.5)
	plt.show()
