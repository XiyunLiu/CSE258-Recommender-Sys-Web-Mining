
# coding: utf-8

# In[118]:

import urllib
import scipy.optimize
import random
import numpy as np
import csv
from sklearn.metrics import mean_squared_error
from sklearn import svm


# In[2]:

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
winedata = list(parseData("http://jmcauley.ucsd.edu/cse255/data/beer/beer_50000.json"))
print "done"


# In[3]:

data[0].keys()


# In[4]:

# y = \theta_0 + \theta_1 * X
year = np.array([review['review/timeStruct']['year'] for review in data])
y = np.array([review['review/overall'] for review in data])

X = np.vstack([np.ones(len(year)), year]).T

theta, residuals, rank, s = np.linalg.lstsq(X, y)

MSE = residuals/len(year)

print theta, MSE


# In[5]:

mean_squared_error(np.dot(X,theta),y)


# In[228]:

X_2 = np.vstack([np.ones(len(year)), year, year**2]).T


# In[229]:

theta_2,residuals_2, rank_2, s_2 = np.linalg.lstsq(X_2, y)


# In[230]:

MSE_2 = residuals_2/len(year)


# In[231]:

print theta_2, MSE_2


# In[232]:

mean_squared_error(np.dot(X_2,theta_2),y)


# # Problem 3

# In[235]:

with open('winequality-white.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    content = []
    for row in reader:
        content.append(row)
indexLabel = {}
for index, label in enumerate(content[0]):
    indexLabel[index+1] = label
data = map(lambda row: [1]+[float(x) for x in row], content[1:])
trainingSet = np.array(data[:len(data)/2])
testSet = np.array(data[len(data)/2:])


# In[237]:

theta_wine


# In[238]:

theta_wine, residuals_wine, rank_wine, s_wine = np.linalg.lstsq(trainingSet[:,:-1], trainingSet[:,-1])


# In[239]:

MSE_wine = residuals_wine/len(trainingSet)

MSE_wine


# In[240]:

print mean_squared_error(np.dot(trainingSet[:,:-1],theta_wine), trainingSet[:,-1])
print mean_squared_error(np.dot(testSet[:,:-1],theta_wine), testSet[:,-1])


# # Problem 4

# In[14]:

for i in range(1,12):
    train, test = np.delete(trainingSet, i, axis=1), np.delete(testSet, i, axis=1)
    theta_ablation, residuals_ablation, rank_ablation, s_ablation = np.linalg.lstsq(train[:,:-1], train[:,-1])
    print "Remove", indexLabel[i],", the MSE on test is",mean_squared_error(np.dot(test[:,:-1],theta_ablation), test[:,-1])


# # Problem 5

# In[196]:

trainingData, trainingLabel = trainingSet[:,1:-1], trainingSet[:,-1]
trainingLabel = map(lambda x: 1 if x>5 else 0, trainingLabel)
testData, testLabel = testSet[:,1:-1], testSet[:,-1]
testLabel = map(lambda x: 1 if x>5 else 0, testLabel)


# In[220]:

# Create a support vector classifier object, with regularization parameter C = 1000
clf = svm.SVC(C=0.5)
clf.fit(trainingData, trainingLabel)


# In[221]:

correct = filter(lambda x: x[0] == x[1], zip(clf.predict(trainingData),trainingLabel))
correctRate = len(correct)*1.0/len(trainingLabel)
print correctRate
correct_test = filter(lambda x: x[0] == x[1], zip(clf.predict(testData), testLabel))
correctRate_test = len(correct_test)*1.0/len(testLabel)
print correctRate_test


# 
# # Problem 6

# In[276]:

import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log

def inner(x,y):
    return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
    return 1.0 / (1 + numpy.exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= logit
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k]*theta[k]
    return -loglikelihood

def fprime(theta, X, y, lam):
    dl = numpy.array([0.0]*len(theta))
    for i in range(len(X)):
        dl += (numpy.array(X[i])/(1+exp(numpy.dot(theta, np.array(X[i])))))
        if not y[i]:
            dl -= X[i]
    dl -= 2*lam*numpy.array(theta) 
    # Negate the return value since we're doing grad ient *ascent*
    return numpy.array([-x for x in dl])

# Use a library function to run gradient descent (or you can implement yourself!)
result = scipy.optimize.fmin_l_bfgs_b(f, np.zeros(len(trainingData[0])), fprime, args = (trainingData, trainingLabel, 1))
theta = result[0]
print theta
print "Loss = ", -result[1]


# In[274]:

probability = sigmoid(numpy.dot(testData, theta))
predict = map(lambda prob: 1 if prob >= 0.5 else 0, probability)
print "Accuracy = ", len(filter(lambda x: x[0] == x[1] , zip(predict, testLabel)))*1.0/len(testLabel)

