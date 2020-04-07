# this function is used to train ukf model

import numpy as np

class model(object):
    def __init__(self):
        __A = 0
        __H = 0
        __Q = 0
        __R = 0
        __P = 0
    def getmodel(self):
        return self.__A, self.__H, self.__Q, self.__R, self.__P
    
    def setmodel(self, A, H, Q, R, P):
        self.__A = A
        self.__H = H
        self.__Q = Q
        self.__R = R
        self.__P = P

class ukf(object):
    ukfModel = 0
    def trainukf(self, x, y):
        self.ukfModel = model
        [m, t] = np.shape(x)
        [n, t] = np.shape(y)
        ypre = y[:, 0:-1]
        ypos = y[:, 1:]
        A = np.dot(np.dot(ypos, ypre.T), np.linalg.pinv(np.dot(ypre, ypre.T)))
        H = np.dot(x, y.T) * np.linalg.pinv(np.dot(y, y.T))
        Q = np.dot(ypos - np.dot(A, ypre), (ypos - A * ypre).T) / (t - 1)
        R = np.dot(x - np.dot(H, y), (x - np.dot(H, y)).T) / t
        P = np.dot(y, y.T) / t
        self.ukfModel.setmodel(self.ukfModel, A, H, Q, R, P)
        return self.ukfModel

    def testukf(self, x):
        [A, H, Q, R, P] = self.ukfModel.getmodel(model) 
        L = 1
        alpha = 1
        kalpha = 0
        beta = 2
        lamda = 3 - L
        # y_t = np.vstack((y, np.sqrt(y[1, :]**2 + y[2, :]**2)))
        # Calculate the UT transform 
        Wm = np.zeros([1, 2*L+1])
        Wc = np.zeros([1, 2*L+1])
        for i in range(2*L+1):
            Wm[0, i] = 1 / (2 * ( L + lamda))
            Wc[0, i] = 1 / (2 * (L + lamda))
        Wm[0] = lamda / (L+lamda)
        Wc[0] = lamda / (L+lamda) + 1 - alpha**2 + beta
        [unitNum, t] = np.shape(x)
        dim = len(A)
        Xukf = np.zeros([dim, t])
        # initial of prediction
        Xukf[:, 0] = 0 
        P0 = np.eye(dim)
        for i in range(1, t):
            xEstimate = Xukf[:, i-1]
            P = P0
            # 1, get the sigma point dataset
            cho = (np.linalg.cholesky(P*(L+lamda))).T
            xgamaP1 = np.zeros([dim, L])
            xgamaP2 = np.zeros([dim, L])
            for k in range(L):
                xgamaP1[:, k] = xEstimate + cho[:, k]
                xgamaP2[:, k] = xEstimate - cho[:, k]
            xSigma = [xEstimate, xgamaP1, xgamaP2]
            # 2, use the sigma point dataset to predict
            XSigmaPre = A * xSigma
            XSigmaPre = XSigmaPre.T
            # 3, Calculate the mean and std error
            Xpred = np.zeros([dim, 1])
            for k in range(2*L+1):
                Xpred = Xpred + Wm[0, k] * XSigmaPre[:, k]
            Ppred = np.zeros([dim, dim])
            for k in range(2*L+1):
                Ppred = Ppred + Wc[0, k] * (XSigmaPre[:, k] - Xpred) * (XSigmaPre[:, k] - Xpred).T
            Ppred = Ppred + Q
            # 4, accroding to the prediction, use UT transform, get the new sigma point dataset
            chor = (np.linalg.cholesky((L*lamda)*Ppred)).T
            XaugSigmaP1 = np.zeros([dim, L])
            XaugSigmaP2 = np.zeros([dim, L])
            for k in range(L):
                XaugSigmaP1[:, k] = Xpred + chor[:, k]
                XaugSigmaP2[:, k] = Xpred - chor[:, k]
            # XaugSigma = [Xpred, XaugSigmaP1, XaugSigmaP2]
            XaugSigma = np.c_[XaugSigmaP1, XaugSigmaP2]
            XaugSigma = np.c_[Xpred, XaugSigma]
            ZsigmaPre = H * XaugSigma
            Zpred = 0
            for k in range(2*L+1):
                Zpred = Zpred + Wm[0, k] *  ZsigmaPre[:, k]
            Pzz = 0
            for k in range(2*L+1):
                # Pzz = Pzz + Wc[0, k] * (ZsigmaPre[:, k] - Zpred) * (ZsigmaPre[:, k] - Zpred).T
                Ztemp = ZsigmaPre[:, k] - Zpred
                Pzz = Pzz + Wc[0, k] * np.dot(Ztemp[:, None], Ztemp[:, None].T)
            Pzz = Pzz + R
            Pxz = np.zeros([dim, unitNum])
            for k in range(2*L+1):
                Pxz = Pxz + Wc[0, k] * (XSigmaPre[:, k] - Xpred) * (ZsigmaPre[:, k] - Zpred).T
            # K = Pxz * np.linalg.pinv(Pzz)
            K = np.dot(Pxz, np.linalg.pinv(Pzz))
            # xEstimate = Xpred + K * (x[:, i] - Zpred)
            xEstimate = Xpred + np.dot(K, x[:, i] - Zpred)
            # P = Ppred - K * Pzz * K.T
            P = Ppred - np.dot(K, np.dot(Pzz, K.T))
            P0 = P
            Xukf[:, i] = xEstimate
        prediction = Xukf[0:1, :]
        return prediction

