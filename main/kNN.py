from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from startEvaluation import *
from getData import *

cross_validation_runs = 10
xNN = 3

training_eval_index = get_cross_validation_file_indices(cross_validation_runs)

mean_accuracy = 0
confusion_matrix = 0
for i in range(cross_validation_runs):
	training_eval_instances = get_training_eval_set(training_eval_index[i])
	training_instances = training_eval_instances[0]
	x = training_instances[0]
	y = training_instances[1]
	
	model = KNeighborsClassifier(n_neighbors=xNN)
	model.fit(x, y)

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