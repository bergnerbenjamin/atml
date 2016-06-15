import numpy as np
#import dataModel as dm
#data = dm.dataModel()
import csv
import math
import pandas as pd

def get_instances_from_csv(datapath, train_or_eval, numrows=50000):
	"""
	Input: datapath: datapath to be read from
	Output: list with first element np array with features, second element np array with respective class labels
	"""
	features = np.zeros(6)
	class_labels = np.zeros(1)
	
	df=pd.read_csv(datapath, sep=',',header=None)
	if train_or_eval == "train":
		features = df.values[1:numrows, 0:6]
		class_labels = df.values[1:numrows, 6]
	else:
		features = df.values[numrows:, 0:6]
		class_labels = df.values[numrows:, 6]
	
	return [features.astype(np.float), class_labels.astype(np.int)]

def getLengthOfFile(datapath):
	with open(datapath, 'r') as f:
		reader = csv.reader(f)
		length_reader = sum(1 for row in reader)
		return length_reader
