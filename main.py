import numpy as np 
import os
import sys
from ReadTrainPath import ReadTrainPath
sys.path.append(r'F:\\PythonProject\\Algorithm\\Kalman')
from Kalman.kalman import kalman
trainPath = os.path.abspath('.') + '\\dataset\\train.txt'
testPath = os.path.abspath('.') + '\\dataset\\test.txt'

x, y = ReadTrainPath(trainPath)
xDim, dataLen = np.shape(x)
yDim, dataLen = np.shape(y)
trainIdx = int(xDim * 0.8)
xTrain = x[:, 0:trainIdx]
yTrain = y[:, 0:trainIdx]
xTest = x[:, trainIdx+1:]
yTest = y[:, trainIdx+1:]
kfmodel = kalman()
kfmodel.trainkf(xTrain, yTrain)
prediction = kfmodel.testkf(xTest, yTest)
cc = np.zeros([1, yDim])
for i in range(0, yDim):
    temp = np.corrcoef(prediction[i, :], yTest[i, :])
    cc[0, i] = temp[0, 1]
print("cc: ", cc)