import numpy as np
def ReadTrainPath(dpath):
    f = open(dpath, 'r')
    argLine = f.readline()
    argLine = argLine.split('\t')
    argNum = len(argLine)
    x = np.zeros([argNum - 1, 100000])
    y = np.zeros([1, 100000])
    temp = f.readlines()
    lineNo = 0
    for line in temp:
        t = line.split('\t')
        for i in range(0, argNum - 1):
            x[i, lineNo] = float(t[i])
        tLast = t[-1]
        y[0, lineNo] = float(tLast[0:-1])
        lineNo = lineNo + 1
    x = x[:, 0:lineNo]
    y = y[:, 0:lineNo]
    return x, y