#!/usr/bin/python

from sklearn import svm, datasets # datasets for testing
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target

svc = svm.SVC().fit(X, y)
print(svc.score)
print(svc.predict(X[1]))
