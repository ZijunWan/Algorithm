import numpy as np 
import os
import sys
from matplotlib import pyplot as plt
from ReadData import ReadData
from Kalman.kalman import kalman
from UKF.ukf import ukf


def get_f(x, y):
    l = x.shape[1]
    R = np.dot(x, x.T) / l
    P = np.dot(x, y.T) / l
    A = np.dot(np.linalg.pinv(R), P).T
    y_est = np.dot(A, x)
    err = y - y_est
    f = lambda x: np.dot(A, x)
    Q = np.dot(err, err.T) / l
    return f, Q


def get_g(x, y):
    l = x.shape[1]
    R = np.dot(y, y.T) / l
    P = np.dot(y, x.T) / l
    H = np.dot(np.linalg.pinv(R), P).T
    x_est = np.dot(H, y)
    err = x - x_est
    g = lambda x: np.dot(H, x)
    R = np.dot(err, err.T) / l
    return g, R



if os.name == 'posix':
    trainPath = os.path.abspath('.') + '/Data/train.txt'
    testPath = os.path.abspath('.') + '/Data/test.txt'
elif os.name == 'nt':
    trainPath = os.path.abspath('.') + '\\Data\\train.txt'
    testPath = os.path.abspath('.') + '\\Data\\test.txt'
    
TrainX, TrainY = ReadData(trainPath, 'train')
TestX = ReadData(testPath, 'test')

# two types of Test DataSet
TrainX = TrainX[:, 0:-500]
TrainY = TrainY[:, 0:-500]
TestX = TrainX[:, -500:]
TestY = TrainY[:, -500:]

xTrainDim, TrainLen = np.shape(TrainX)
yTrainDim, TrainLen = np.shape(TrainY)
xTestDim, TestLen = np.shape(TestX)


# kalman filter
cc = np.zeros([1, yTrainDim])
# kfModel = kalman()
# kfModel.trainkf(TrainX, TrainY)
# predictionKF = kfModel.testkf(TestX)
# for i in range(0, yTrainDim):
#     temp = np.corrcoef(predictionKF[i, :], TestY[i, :])
#     cc[0, i] = temp[0, 1]
# print("cc: ", cc)

# unscented kalman filter
ukfModel = ukf(get_f, get_g)
ukfModel.trainukf(TrainX, TrainY)
predictionUKF = ukfModel.testukf(TestX)
for i in range(0, yTrainDim):
  temp = np.corrcoef(predictionUKF[i, :], TestY[i, :])
  cc[0, i] = temp[0, 1]
print("CC-UKF: ", cc)

# plot the result
# pltX = range(len(predictionUKF[0, :]))
# plt.plot(pltX, predictionUKF[0, :])
# plt.show()