# Wiener Filter

import numpy as np 

class WienerFilter(object):
    def trainWF(self, x, y):
        [xDim, xLen] = np.shape(x)
        [yDim, yLen] = np.shape(y)
        if xLen != yLen:
            raise ValueError('dimension of x and y is not the same')
        
    

    def LeastSquare(self, x, y):
        # x: k*t, k: dimension, t: time
        # y: m*t, m: dimension, t:time
        # calculate the regression coefficient
        [xDim, xLen] = np.shape(x)
        [yDim, yLen] = np.shape(y)
        x = np.vstack((x, np.ones([1, xLen])))
        Xcor = np.linalg.inv(np.dot(x, x.T))
        XYCor = np.dot(x.T, y)
        W = np.dot(Xcor, XYCor)
        Err = y - np.dot(W, x)
        return W, Err
    
    def CalcSingleDimH(self, x, y):
        [xDim, xLen] = np.shape(x)
        [yDim, yLen] = np.shape(y)
        if xDim != 1 or yDim != 1:
            raise ValueError('x or y is not a vector')
        [W, Err] = self.LeastSquare(x, y)
        Rxx = np.correlate(x, x)
        RxxMatrix = np.zeros([xLen, xLen])
        for i in range(xLen):
            for j in range(xLen):
                RxxMatrix[i, j] = Rxx[0, np.abs(i-j)]
        Rxs = np.correlate(x, Err)
        h = np.dot(np.linalg.inv(Rxx), Rxs)
        