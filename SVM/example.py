# -*- coding: utf-8 -*-

from sklearn import datasets
from SupportVectorMachine import SupportVectorMachine
import numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target

svmModel = SupportVectorMachine(1.0, 'poly', 0.0001)
dataLen = len(x)
trainIdx = int(np.floor(dataLen * 0.7))
xTrain = x[0 : trainIdx, :]
yTrain = y[0 : trainIdx]
xTest = x[trainIdx + 1 : dataLen, :]
yTest = y[trainIdx + 1 : dataLen]

svmModel.trainsvm(xTrain, yTrain)
score = svmModel.testsvm(xTest, yTest)
print(score)
