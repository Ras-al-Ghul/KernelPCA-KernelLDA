import numpy as np
from copy import deepcopy
import random, math, sys
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

numoffeatures = 10000
newnumoffeatures = 100
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

	return np.column_stack((eigvecs[:,-i] for i in range(1, newnumoffeatures+1))) 

if __name__ == '__main__':
	random.seed()

	data = np.zeros((numofdata, numoffeatures))
	populatedata(data)

	labels = getlabels('arcene_train.labels', numofdata)

	newdata = kernelpca(data, 0.2)

	# plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == 1], [newdata[i][1] for i in range(len(newdata)) if labels[i] == 1], color='red', alpha=0.5)
	# plt.scatter([newdata[i][0] for i in range(len(newdata)) if labels[i] == -1], [newdata[i][1] for i in range(len(newdata)) if labels[i] == -1], color='blue', alpha=0.5)
	# plt.show()
