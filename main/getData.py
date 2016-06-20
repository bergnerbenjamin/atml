import numpy as np
import pandas as pd
import csv

def get_cross_validation_file_indices(cross_validation_runs):
    init = [e for e in range(10)]
    training_eval = []
    for i in range(cross_validation_runs):
        init_temp = init[:]
        eval = init_temp[i]
        del init_temp[i]
        training = init_temp
        training_eval.append([training, eval])
    #print("training_eval: ", training_eval)
    return training_eval
    
def get_training_eval_set(training_eval_index):
    """
    Input: Number of files for training set
    Output: all training instances
    """
    instances = [np.array([], dtype=np.float32), np.array([], dtype=np.float32)]
    for i in training_eval_index[0]:
        DATAPATH = "../sub_datasets/subset_"+str(i)+".csv"
        file_instances = get_instances_from_csv(DATAPATH, "all")
        if instances[0].size == 0 and instances[1].size == 0:
            instances_train = file_instances
        else:
            instances_train = [np.vstack((instances[0], file_instances[0])), np.append(instances[1], file_instances[1])]
    
    DATAPATH = "../sub_datasets/subset_"+str(training_eval_index[1])+".csv"
    instances_eval = get_instances_from_csv(DATAPATH, "all")
    
    return [instances_train, instances_eval]

def get_instances_from_csv(datapath, train_or_eval):
	"""
	Input: datapath: datapath to be read from
	Output: list with first element np array with features, second element np array with respective class labels
	"""
	features = np.zeros(6)
	class_labels = np.zeros(1)
	
	df = pd.read_csv(datapath, sep=',',header=None)[1:]
	df = df.sample(frac=1).reset_index(drop=True)
	if train_or_eval == "train":
		features = df.values[1:500000, 0:6]
		class_labels = df.values[1:500000, 6]
	elif train_or_eval == "eval":
		features = df.values[500000:, 0:6]
		class_labels = df.values[500000:, 6]
	else:
		features = df.values[1:, 0:6]
		class_labels = df.values[1:, 6]
	
	#print("features: ", features)
	#print("class_labels: ", class_labels)
	
	return [features.astype(np.float), class_labels.astype(np.int)]