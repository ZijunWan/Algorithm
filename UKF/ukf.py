# this function is used to train ukf model

import numpy as np

class ukf():
    def __init__(self, get_f, get_g):
        self.ukf_model = dict()
        self.get_f = get_f
        self.get_g = get_g


    def trainukf(self, x, y):
        [m, t] = np.shape(x)
        [n, t] = np.shape(y)
        f, Q = self.get_f(y[:, 0:-1], y[:, 1:])
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
        kai = 1
        alpha = 1
        beta = 1
        p_lambda = alpha**2 * (n + kai) - n

        Wm = np.zeros([1, 2*n+1])
        Wc = np.zeros([1, 2*n+1])
        for i in range(2*n+1):
            Wm[0, i] = 1 / (2 * (n + p_lambda))
            Wc[0, i] = 1 / (2 * (n + p_lambda))
        Wm[0, 0] = p_lambda / (n+p_lambda)
        Wc[0, 0] = p_lambda / (n+p_lambda) + 1 - alpha**2 + beta
        # [unitNum, t] = np.shape(x)
        # dim = len(A)
        y_ukf = np.zeros([n, t])
        # initial of prediction
        y_ukf[:, 0] = 0 
        P0 = np.eye(n)
        for i in range(1, t):
            y_est = y_ukf[:, i-1, None]
            P = P0
            # 1, get the sigma point dataset
            cho = np.linalg.cholesky(P*(n+p_lambda))
            y_p1 = np.zeros([n, n])
            y_p2 = np.zeros([n, n])
            for k in range(n):
                y_p1[:, k] = y_est + cho[:, k]
                y_p2[:, k] = y_est - cho[:, k]
            y_sigma = np.c_[y_est, y_p1, y_p2]
            # 2, use the sigma point dataset to predict
            y_sigma_pred = np.zeros([n, 2*n+1])
            y_mu = np.zeros([1, n])
            for k in range(2*n+1):
                y_sigma_pred[:, k] = A(y_sigma[:, k])
                y_mu += Wm[0, k] * y_sigma_pred[:, k, None]
            
            P_x = Q
            for k in range(2*n+1):
                P_x += Wc[0, k] * np.dot((y_sigma_pred[:, k, None] - y_mu), (y_sigma_pred[:, k, None] - y_mu).T)
            chol_pos = np.linalg.cholesky((n+p_lambda)*P_x)

            for k in range(n):
                y_p1[:, k] = y_mu + chol_pos[:, k]
                y_p2[:, k] = y_mu - chol_pos[:, k]
            y_pos = np.c_[y_mu, y_p1, y_p2]

            z_sigma = np.zeros([m, 2*n+1])
            for k in range(2*n+1):
                z_sigma[:, k, None] = H(y_pos[:, k, None])
            z_mu = np.zeros([m, 1])
            for k in range(2*n+1):
                z_mu += Wm[0, k] * z_sigma[:, k, None]
            P_z = R
            for k in range(2*n+1):
                P_z += Wc[0, k] * np.dot((z_sigma[:, k, None] - z_mu), (z_sigma[:, k, None] - z_mu).T)
            
            P_xz = np.zeros([n, m])
            for k in range(2*n+1):
                P_xz += Wc[0, k] * np.dot((y_pos[:, k, None] - y_mu), (z_sigma[:, k, None] - z_mu).T)
            
            K = np.dot(P_xz, np.linalg.pinv(P_z))
            y_ukf[:, i, None] = y_mu + np.dot(K, x[:, i, None] - z_mu)
            P0 = P_x - np.dot(np.dot(K, P_z), K.T)
        return y_ukf

