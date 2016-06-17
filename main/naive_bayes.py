from sklearn.naive_bayes import GaussianNB
from startEvaluation import *
from getData import *

def onlineNaiveBayes(batch_size):	
	j = 0
	while True:
		if j == 0:
			model.partial_fit(x[j:j+batch_size], y[j:j+batch_size], y)
		else:
			try:
				model.partial_fit(x[j:j+batch_size], y[j:j+batch_size])
			except:
				break
		j += batch_size

def offlineNaiveBayes():
	model.fit(x, y.ravel())

training_range = 9
test_index = 9
batch_size = 100

instances = get_training_set(training_range)	
x = instances[0]
y = instances[1]

model = GaussianNB()
onlineNaiveBayes(batch_size)
#offlineNaiveBayes()

DATAPATH = "../sub_datasets/subset_9.csv"
eval_instances = get_instances_from_csv(DATAPATH, "all", False)
eval_x = eval_instances[0]
eval_y = eval_instances[1]

predicted = model.predict(eval_x)
print("predicted_real: ", predicted[0:50])
print("groundtruth: ", eval_y[0:50].ravel())

test_eval = evaluation(eval_y.ravel(), predicted)
test_eval.print_eval()