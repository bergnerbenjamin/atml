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

cross_validation_runs = 10
batch_size = 100

training_eval_index = get_cross_validation_file_indices(cross_validation_runs)

mean_accuracy = 0
confusion_matrix = 0
for i in range(cross_validation_runs):
	training_eval_instances = get_training_eval_set(training_eval_index[i])
	training_instances = training_eval_instances[0]
	x = training_instances[0]
	y = training_instances[1]

	model = GaussianNB()
	#onlineNaiveBayes(batch_size)
	offlineNaiveBayes()

	eval_instances = training_eval_instances[1]
	eval_x = eval_instances[0]
	eval_y = eval_instances[1]
	predicted = model.predict(eval_x)
	print("predicted_real: ", predicted[0:50])
	print("groundtruth: ", eval_y[0:50].ravel())
	
	test_eval = evaluation(eval_y.ravel(), predicted)
	test_eval.print_eval()
	mean_accuracy += test_eval.get_accuracy()
	confusion_matrix += test_eval.get_conf_matrix()
mean_accuracy /= cross_validation_runs
print("mean_accuracy: ", mean_accuracy)
print("confusion_matrix: ", confusion_matrix)