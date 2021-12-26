# this function is used to train ukf model

import numpy as np

class ukf():
    def __init__(self, get_f, get_g):
        self.ukf_model = dict()
        self.get_f = get_f
        self.get_g = get_g


    def trainukf(self, x, y):
        _, xt = np.shape(x)
        _, yt = np.shape(y)
        if xt != yt:
            raise(ValueError, "Length of TrainX and TrainY is not the same")
        t = xt
        del xt, yt
        # Q = np.dot(self.f(y[:, 0:-1])-y[:, 1:], (self.f(y[:, 0:-1])-y[:, 1:]).T) / (t-1)
        # R = np.dot(self.g(x)-y, (self.g(x)-y).T) / t
        f, Q = self.get_f(y[:, :-1], y[:, 1:])
        g, R = self.get_g(x, y)
        self.ukf_model['A'] = lambda x: f(x)
        self.ukf_model['H'] = lambda x: g(x)
        self.ukf_model['Q'] = Q
        self.ukf_model['R'] = R
        self.ukf_model['P'] = np.dot(y, y.T) / t
        return self.ukf_model

    def testukf(self, x):
        A = self.ukf_model['A']
        H = self.ukf_model['H']
        Q = self.ukf_model['Q']
        R = self.ukf_model['R']
        P = self.ukf_model['P']
        
        m, t = x.shape
        n, _ = Q.shape
        kai = 0.7 # 0 or 3 - n, n is the dimension of y
        alpha = 0.9 # 0 < alpha <=1
        beta = 2 # 2 for gaussian distribution
        p_lambda = alpha**2 * (n + kai) - n

        Wm = np.zeros(2*n+1)
        Wc = np.zeros(2*n+1)
        for i in range(1, 2*n+1):
            Wm[i] = 1 / (2 * (n + p_lambda))
            Wc[i] = 1 / (2 * (n + p_lambda))
        Wm[0] = p_lambda / (n+p_lambda)
        Wc[0] = p_lambda / (n+p_lambda) + 1 - alpha**2 + beta
        # [unitNum, t] = np.shape(x)
        # dim = len(A)
        y_ukf = np.zeros([n, t])
        # initial of prediction
        P = np.eye(n)
        for i in range(1, t):
            y_est = y_ukf[:, i-1, None]
            # 1, get the sigma point dataset
            # cho = np.linalg.cholesky(P*(n+p_lambda))
            # Use SVD instead of cholesky
            s, v, _ = np.linalg.svd(P*(n+p_lambda))
            cho = np.dot(s, np.sqrt(np.diag(v)))

            y_p1 = np.zeros([n, n])
            y_p2 = np.zeros([n, n])
            for k in range(n):
                y_p1[:, k] = y_est + cho[:, k]
                y_p2[:, k] = y_est - cho[:, k]
            y_sigma = np.c_[y_est, y_p1, y_p2]
            # 2, use the sigma point dataset to predict
            y_sigma_pred = A(y_sigma)
            y_mu = np.zeros([1, n])
            for k in range(2*n+1):
                y_mu += Wm[k] * y_sigma_pred[:, k, None]
            
            P_x = Q
            for k in range(2*n+1):
                P_x += Wc[k] * np.dot((y_sigma_pred[:, k, None] - y_mu), (y_sigma_pred[:, k, None] - y_mu).T)
            
            # chol_pos = np.linalg.cholesky((n+p_lambda)*P_x)
            # Use SVD instead of cholesky to avoid non-positive definite matrix
            s, v, _ = np.linalg.svd((n+p_lambda)*P_x)
            chol_pos = np.dot(s, np.sqrt(np.diag(v)))
            for k in range(n):
                y_p1[:, k] = y_mu + chol_pos[:, k]
                y_p2[:, k] = y_mu - chol_pos[:, k]
            y_pos = np.c_[y_mu, y_p1, y_p2]

            z_sigma = H(y_pos)
            z_mu = np.zeros([m, 1])
            for k in range(2*n+1):
                z_mu += Wm[k] * z_sigma[:, k, None]
            P_zz = R
            for k in range(2*n+1):
                P_zz += Wc[k] * np.dot((z_sigma[:, k, None] - z_mu), (z_sigma[:, k, None] - z_mu).T)
            
            P_xz = np.zeros([n, m])
            for k in range(2*n+1):
                P_xz += Wc[k] * np.dot((y_pos[:, k, None] - y_mu), (z_sigma[:, k, None] - z_mu).T)
            
            K = np.dot(P_xz, np.linalg.pinv(P_zz))
            y_ukf[:, i, None] = y_mu + np.dot(K, x[:, i, None] - z_mu)
            P = P_x - np.dot(np.dot(K, P_zz), K.T)
        return y_ukf

