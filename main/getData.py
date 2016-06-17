import numpy as np
import pandas as pd
import csv

def get_training_set(training_range, silent=False):
    """
    Input: Number of files for training set
    Output: all training instances
    """
    instances = [np.array([], dtype=np.float32), np.array([], dtype=np.float32)]
    for i in range(training_range):
        DATAPATH = "../sub_datasets/subset_"+str(i)+".csv"
        file_instances = get_instances_from_csv(DATAPATH, "all", False, silent=silent)    
        if instances[0].size == 0 and instances[1].size == 0:
            instances = file_instances
        else:
            instances = [np.vstack((instances[0], file_instances[0])), np.append(instances[1], file_instances[1])]
    return instances

def get_instances_from_csv(datapath, train_or_eval, fixed, silent=False):
    """
    Input: datapath: datapath to be read from
    Output: list with first element np array with features, second element np array with respective class labels
    """
    features = np.zeros(6)
    class_labels = np.zeros(1)
    
    df=pd.read_csv(datapath, sep=',',header=None)
    if train_or_eval == "train":
        features = df.values[1:500000, 0:6]
        class_labels = df.values[1:500000, 6]
    elif train_or_eval == "eval":
        features = df.values[500000:, 0:6]
        class_labels = df.values[500000:, 6]
    else:
        features = df.values[1:, 0:6]
        class_labels = df.values[1:, 6]
    
    if not silent:
        print("features: ", features)
        print("class_labels: ", class_labels)
    
    return [features.astype(np.float), class_labels.astype(np.int)]
