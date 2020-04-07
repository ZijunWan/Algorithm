import numpy as np
def ReadTrainPath(dpath, type):
    # dpath: data path
    # type: train data or test data
    if type != 'train' and type != 'test':
        raise ValueError("unknown type")
    f = open(dpath, 'r')
    argLine = f.readline()
    argLine = argLine.split('\t')
    argNum = len(argLine)
    temp = f.readlines()
    lineLen = len(temp)
    if type == 'train': 
        x = np.zeros([argNum - 1, lineLen])
        y = np.zeros([1, lineLen])
    else :
        x = np.zeros([argNum, lineLen])
    lineNo = 0
    for line in temp:
        t = line.split('\t')
        for i in range(0, argNum - 1):
            x[i, lineNo] = float(t[i])
        if type == 'train': 
            tLast = t[-1]
            y[0, lineNo] = float(tLast[0:-1])
        lineNo = lineNo + 1
    if type == 'train': 
        x = x[:, 0:lineNo]
        y = y[:, 0:lineNo]
        return x, y
    else :
        x = x[:, 0:lineNo]
        return x