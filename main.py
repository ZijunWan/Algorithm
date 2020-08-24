import numpy as np 
import os
import sys
from matplotlib import pyplot as plt
from ReadData import ReadData
from Kalman.kalman import kalman
from UKF.ukf import ukf
if os.name=='posix':
    trainPath = os.path.abspath('.') + '/Data/train.txt'
    testPath = os.path.abspath('.') + '/Data/test.txt'
elif os.name=='nt':
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
kfModel = kalman()
kfModel.trainkf(TrainX, TrainY)
predictionKF = kfModel.testkf(TestX)

# unscented kalman filter
ukfModel = ukf()
ukfModel.trainukf(TrainX, TrainY)
predictionUKF = ukfModel.testukf(TestX)
cc = np.zeros([1, yTrainDim])
for i in range(0, yTrainDim):
    temp = np.corrcoef(predictionKF[i, :], TestY[i, :])
    cc[0, i] = temp[0, 1]
print("cc: ", cc)

# plot the result
pltX = range(len(predictionUKF[0, :]))
plt.plot(pltX, predictionUKF[0, :])
plt.show()