from sklearn.naive_bayes import GaussianNB
import numpy as np
import csv
from startEvaluation import *
import pandas as pd

DATAPATH = "../sub_datasets/subset_0.csv"
DATAPATH1 = "../sub_datasets/subset_5.csv"

def get_instances_from_csv(datapath, train_or_eval, fixed, length_of_file):
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
	else:
		features = df.values[500000:, 0:6]
		class_labels = df.values[500000:, 6]
	
	print("features: ", features)
	print("class_labels: ", class_labels)
	
	return [features.astype(np.float), class_labels.astype(np.int)]
	
	# with open(datapath, 'r') as f:
		# reader = csv.reader(f)
		# i = 0
		# if fixed:
			# begin_row = 0
			# end_row = 5
		# else:
			# begin_row = 1 if train_or_eval == "train" else round(0.99 * length_of_file)
			# end_row = round(0.05 * length_of_file) if train_or_eval == "train" else length_of_file
		
		# print("begin_row: ", begin_row)
		# print("end_row: ", end_row)
		
		# for row in reader:
			# if i > begin_row:
				# features = np.vstack([features, row[0:6]])
				# class_labels = np.append(class_labels, row[-1])
			# if i > end_row:
				# break
			# i += 1
		# features = np.delete(features, 0)
	# return [features.astype(np.float), class_labels.astype(np.float)]

def getLengthOfFile(datapath):
	with open(datapath, 'r') as f:
		reader = csv.reader(f)
		length_reader = sum(1 for row in reader)
		return length_reader
	
def getClassInstances(instances, classes, class_id):
	return instances[np.where(classes == class_id)[0]]

def getClassMeanOfFeatures(instances_of_class):
	return np.mean(instances_of_class,0)

def getClassVarianceOfFeatures(instances_of_class):
	return np.var(instances_of_class,0)

def constructStatsForBayes(instances, classes):
	for i in range(classes.size):
		instances_of_class = getClassInstances(instances, classes, classes[i])
		class_mean_of_features = getClassMeanOfFeatures(instances_of_class)
		class_var_of_features = getClassVarianceOfFeatures(instances_of_class)
		if i == 0:
			theta = class_mean_of_features
			sigma = class_var_of_features
		else:
			theta = np.vstack((theta, class_mean_of_features))
			sigma = np.vstack((sigma, class_var_of_features))
	return (theta,sigma)

length_of_file = getLengthOfFile(DATAPATH)
instances = get_instances_from_csv(DATAPATH1, "train", False, length_of_file)
#print("instances: ", instances)

#assigning predictor and target variables
x = instances[0]
y = instances[1]

print("x: ", x)
print("y: ", y)

#theta, sigma = constructStatsForBayes(x, y)

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y.ravel())

#Predict Output 
#predicted= model.predict([[-4.9667664, -1.8327484, 10.064636, -0.10971069, 0.048919678, 0.16738892]])
#print("predicted: ", predicted)

eval_instances = get_instances_from_csv(DATAPATH1, "eval", False, length_of_file)
eval_x = eval_instances[0]
eval_y = eval_instances[1]

predicted = model.predict(eval_x)
print("predicted_real: ", predicted[0:50])
print("groundtruth: ", eval_y[0:50].ravel())

test_eval = evaluation(eval_y.ravel(), predicted)
#print all generated data
test_eval.print_eval()