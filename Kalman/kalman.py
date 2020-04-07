# Kalman Filter

import numpy as np

class kalman(object):
    A = 0
    H = 0
    Q = 0
    W = 0
    def trainkf(self, x, y):
        [m, t] = np.shape(x)
        [n, t] = np.shape(y)
        ypre = y[:, 0:-1]
        ypos = y[:, 1:]
        self.A = np.dot(np.dot(ypos, ypre.T), np.linalg.pinv(np.dot(ypre, ypre.T)))
        self.H = np.dot(x, y.T) * np.linalg.pinv(np.dot(y, y.T))
        self.W = np.dot(ypos - np.dot(self.A, ypre), (ypos - self.A * ypre).T) / (t - 1)
        self.Q = np.dot(x - np.dot(self.H, y), (x - np.dot(self.H, y)).T) / t
        self.P = np.dot(y, y.T) / t
    
    def testkf(self, x):
        [m, t] = np.shape(x)
        n = np.shape(self.A) 
        prediction = np.zeros([n[0], t])
        P = self.P
        for i in range(1, t):
            x_m = np.dot(self.A, prediction[:, i-1])
            P_m = np.dot(np.dot(self.A, P), self.A.T) + self.W
            temp = np.dot(np.dot(self.H, P_m), self.H.T) + self.Q
            K = P_m * np.dot(self.H.T, np.linalg.pinv(temp))
            prediction[:, i] = x_m + np.dot(K, x[:, i] - np.dot(self.H, x_m))
            P = P_m - np.dot(np.dot(K, self.H), P_m)
        return prediction
