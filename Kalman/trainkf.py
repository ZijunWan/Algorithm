# this function is used to train kalam model
import numpy as np 

class Model(object):
    def __init__(self, A, H, W, Q):
        self.A = A
        self.H = H
        self.W = W
        self.Q = Q

def trainkf(x, y):
    [m, t] = np.shape(x)
    [n, t] = np.shape(y)
    ypre = y[:, 0:-1]
    ypos = y[:, 1:]
    A = np.dot(np.dot(ypos, ypre.T), np.linalg.pinv(np.dot(ypre, ypre.T)))
    H = np.dot(x, y.T) * np.linalg.pinv(np.dot(y, y.T))
    W = np.dot(ypos - np.dot(A, ypre), (ypos - A * ypre).T) / (t - 1)
    Q = np.dot(x - np.dot(H, y), (x - np.dot(H, y)).T) / t
    model = Model(A, H, W, Q)
    return model 
