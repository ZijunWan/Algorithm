# -*- coding: utf8 -*-

from sklearn import svm
import numpy as np

class SupportVectorMachine(object):
    def __init__(self, C, kernel, tol=0.001):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        
    def trainsvm(self, x, y):
        self.model = svm.SVC(C=self.C, kernel=self.kernel, degree=3, gamma='auto', coef0=0.0, shrinking=True, 
                probability=False, tol=self.tol, cache_size=200, class_weight=None, 
                verbose=False, max_iter=-1, decision_function_shape='ovr', 
                random_state=None)
        self.model.fit(x, y)

    def testsvm(self, x, y):
        prediction = self.model.predict(x)
        return prediction
