# this function is used to get prediction using the Kalman model

def testkf(model, x, y):
    [m, t] = np.shape(x)
    [n, t] = np.shape(y)       
    prediction = np.zeros([n, t])
    P = np.W
    for i in range(1, t):
        x_m = np.dot(model.A, prediction[:, i-1])
        P_m = np.dot(np.dot(model.A, P), model.A.T) + model.W
        temp = np.dot(np.dot(model.H, P_m), model.H.T) + model.Q
        K = P_m * np.dot(model.H.T, np.lianlg.pinv(temp))
        prediction[:, i] = x_m + np.dot(K, x[:, i] - np.dot(model.H, x_m))
        P = P_m - np.dot(np.dot(K, model.H), P_m)
    return prediction