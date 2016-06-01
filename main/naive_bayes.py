from sklearn.naive_bayes import GaussianNB
import numpy as np
#import dataModel as dm
#data = dm.dataModel()
import csv

DATAPATH = "../sub_datasets/subset_0.csv"

def get_instances_from_csv(datapath):
	"""
	Input: datapath: datapath to be read from
	Output: list with first element np array with features, second element np array with respective class labels
	"""
	features = np.zeros(6)
	class_labels = np.zeros(1)
	with open(datapath, 'r') as f:
		reader = csv.reader(f)
		i = 0
		for row in reader:
			if i>0:
				print("np.array(row): ", np.array(row))
				features = np.vstack([features, row[0:6]])
				class_labels = np.append(class_labels, row[-1])
			if i > 5:
				break
			i += 1
	return [features.astype(np.float), class_labels.astype(np.float)]

instances = get_instances_from_csv(DATAPATH)
print("instances: ", instances)

#assigning predictor and target variables
x= instances[0] #np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = instances[1] #np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict([[-4.9667664, -1.8327484, 10.064636, -0.10971069, 0.048919678, 0.16738892]])

print("predicted: ", predicted)

#Output: ([3,4])