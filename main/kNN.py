from sklearn.neighbors import KNeighborsClassifier
from startEvaluation import *
from getData import *
from sklearn import tree

training_range = 9
test_index = 9

instances = get_training_set(training_range)	
x = instances[0]
y = instances[1]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)

DATAPATH = "../sub_datasets/subset_9.csv"
eval_instances = get_instances_from_csv(DATAPATH, "all", False)
eval_x = eval_instances[0]
eval_y = eval_instances[1]

predicted = neigh.predict(eval_x)
print("predicted_real: ", predicted[0:50])
print("groundtruth: ", eval_y[0:50].ravel())

test_eval = evaluation(eval_y.ravel(), predicted)
test_eval.print_eval()