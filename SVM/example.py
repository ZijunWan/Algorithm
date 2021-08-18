# -*- coding: utf-8 -*-

from sklearn import datasets
from SupportVectorMachine import SupportVectorMachine
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

dataLen = len(x)
prediction = np.zeros([dataLen])
svmModel = SupportVectorMachine(1.0, 'poly', 0.0001)
for i in range(dataLen):
    train_x = np.delete(x, i, axis=0)
    train_y = np.delete(y, i)
    test_x = [x[i, :]]
    test_y = y[i]
    svmModel.trainsvm(train_x, train_y)
    prediction[i] = svmModel.testsvm(test_x, test_y)

score = np.shape(np.where(prediction-y == 0))[1] / len(y)
print(score)

