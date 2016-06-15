#!/usr/bin/python

from sklearn import svm, datasets, preprocessing
import numpy as np
from loaddata import get_instances_from_csv
from startEvaluation import evaluation

DATAPATH = "../sub_datasets/subset_0.csv"

training_data = get_instances_from_csv(DATAPATH, "train", numrows=50000)
eval_data = get_instances_from_csv(DATAPATH,"eval", numrows=500000)

kernels = ['poly', 'rbf', 'linear', 'sigmoid']

scaler = preprocessing.StandardScaler()
scaler.fit(training_data[0])

def print_prediction(training_data, eval_data, *args, **kwargs):

    svc = svm.SVC(*args, **kwargs).fit(training_data[0], training_data[1])
    print("accuracy on traning data:")
    ev = evaluation(training_data[1], svc.predict(training_data[0]))
    ev.print_only_accuracy()
    print("accuracy on evaluation data:")
    ev = evaluation(eval_data[1], svc.predict(eval_data[0]))
    ev.print_only_accuracy()

for kernel in kernels:
    print(kernel)
    print_prediction(training_data, eval_data, kernel=kernel)

training_data_scaled = [scale.transform(training_data[0]), training_data[1]]
eval_data_scaled = [scale.transform(eval_data[0]), eval_data[1]]
for kernel in kernels:
    print(kernel)
    print_prediction(training_data_scaled, eval_data_scaled, kernel=kernel)
