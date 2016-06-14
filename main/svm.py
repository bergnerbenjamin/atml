#!/usr/bin/python

from sklearn import svm, datasets # datasets for testing
import numpy as np
from loaddata import get_instances_from_csv
from startEvaluation import evaluation

DATAPATH = "../sub_datasets/subset_0.csv"

data = get_instances_from_csv(DATAPATH, numrows=10000)


def print_prediction_with_svc_params(*args, **kwargs):

    svc = svm.SVC(*args, **kwargs).fit(data[0], data[1])
    ev = evaluation(data[1], svc.predict(data[0]))
    ev.print_eval()

