
# Environment used
Python 3 in general; numpy, pandas, sklearn as modules
* numpy is for convenient and fast vector/matrix handling of data
* pandas for processing csv files
* sklearn enables us to fastly implement machine learning algorithms

IPython Notebooks are used for presenting Code/Documentation/Presentation

# Analyse the data set and the attributes:

We selected accelerometer and gyrometer for x,y and z coordinates because these are the most important informationen to determine whether somebody moves, sits, etc.
We did not use:  
* the timestamp, because it does not represent the application properties and uses a new value for every instance
side information 
* the user because the classificatin should work for all users, irrespective of how the runs/walks,sits,...
* we did not use the information about the smartphone because we want to achieve a high generalization for all

# 3 Approach

Our approach is based on several different algorithms. We were very unfamiliar with the proposed data, to achieve good results anyway we decide to implement four algorithms instead of two. A short description of them is given in the following. Also their abilities to be used as on online or offline variant are mentioned.

## K-Nearest Neighbour Algorithm (KNN)

A very simple classifier, which is based on the k-nearest neighbours of a data point. When a new data point should be classified, all its labeled k nearest neighbours are taken into account. The data point is assigned to the class to which most of its k nearest neighbours are assigned.

The algorithm can be used as online of offline variant. There are no differences between those variants, the result is the same in both cases.

## Support-Vector-Machine SVM

SVMs try to seperate classes in an N-dimensional space with a N-1-dimensional hyperplane. Because the data is not linear separable in most cases, the data is transformed to another space in which it is linear separable. This mapping is done with a kernel function.

SVMs have several different parameters which have to be adapted to achieve a good result. SVMs are an offline classification method, all the data has to be avaibalbe in advance to classify new data points.

## Decition Trees

Decision Trees are a classification method which is based on trees from the field of graph theory. Every node distinguishes the dataset into different subsets. Leaf nodes contain no condition but a class label which is assigned to the data point. To classify a data point, a top-down traversal trough the tree is performed. Which child is selected depends on the condition of the actual node and the attribute value of the data point which has to be assigned. 

The tree has to be created before it can be used to classify new data points, it is an offline approach.

## Naive Bayes

Naive Bayes is a simple classifier which is based on the conditionally independence of all considered attributes. When this assumption is not fulfilled, this classifier leads to bad results.

The algorithm is implemented as online or offline variant.

# Data cleaning

## Overview
The data consists of two data sets, accelerometer and gyroscope data set.
The first step is to merge this two data set into a new, merged data set and to get rid of outliers and missing values.
Therefore the accelerometer data entries have to match corresponding gyroscope data entries.

First, let's look at the accelerometer and gyroscope data attributes (gt is the label):

accelerometer data attributes:     Index,Arrival_Time,Creation_Time,x,y,z,User,Model,Device,gt
gyroscope data attributes:         Index,Arrival_Time,Creation_Time,x,y,z,User,Model,Device,gt

Both data sets contain the same attribute labels.
The Index,Arrival_Time,Creation_Time doesn't have any relevant information content for the classification algorithm because most of them are unique identifiers. These attributes will be ignored in the merge process of the two data sets.
To learn a general model that will handle and classify all devices and users, the attributes User,Model and Device will be ignored.
Thus we have 4 Attributes left in both data sets which needs to be merged: x,y,z,gt

To be not confused with two x, y and z values the three values from accelerometer will be named aX, aY and aZ and from gyroscope gX, gY and gZ respectively. For reason of understanding the attribute name gt will be named label.

The result of the merged data set will be a set of seven attributes: aX,aY,aZ,gX,gY,gZ,label.
Because of the vast data file, the file is splitted in a further step into ten subsets which also will allow an easier cross-validation, where nine of the subsets will be used to train and one to test the classifier.

## Data pruner
The Class DataPruner was implemented to prune and merge the two data sets.
For this purpose we decided to find the two matching entries by the attributes Index and gt (label) of a given entry.
The main problem is that you cannot load the two data sets into the main memory because they're too big (2.48GB both CSV files combined).
To solve this problem the DataPruner method dataPruning uses two CSV reader, one for each data set, which goes through all entries row by row.
The Index and gt (label) attribute will be used to find a matching row because the problem, as mentioned before, of other appropriate attributes Arrival_Time,Creation_Time is the distinct assignability. The difference of the time values in accelerometer and gyroscope is about two and often the time values don't match exactly. With the use of the Index value an exact assignment can be done.
Also the amount of specific labeled data entries differs in the two data sets, and so the last Index values of a set of specific labeled data entries will differ.

The implemented strategy is:
Go through all data entries (rows) in both data sets, accelerometer and gyroscope.

Do while rows exist:
    1. Find the next matching Index entries:
        a. Load next parsable and valid rows accelerometer and gyroscope.
           (Valid means all entries x,y,z as float and gt as string can be parsed)
        b. If the Index values match: go to 2.
        c. If Index value are unequal and one of them is zero:
            Skip rows of the other reader till it have also the value zero and go to 2.
        d. else:
            Skip one row of the reader with the lower Index value and go to 1.b.
    2. Matching rows found:
        a. Read in the x,y,z values of the accelerometer as aX,aY,aZ and x,y,z values of gyroscope as gX,gY,gZ and write it with the corresponding label (gt) into the merged data set.
        b. Go to 1.

Data entries with label (gt) null will be ignored.

# Convert data into a format useable for machine learning algorithms

## Overview
Through the fact that the merged and pruned data set was still too vast to handle, it was splitted into ten subsets. The subsets should reflect the same proportion of data entries with the same label, we first split the data set by label (Pruned data separator) and merge them randomly with respect to the label proportions (Classificator data generator).

## Pruned data separator
The Class PrunedDataSeparator was implemented to split the merged data set into ten subsets.
For each label item one file was created (label file) and then the merged data set was parsed row by row.
For each row the label was determined and copied into the according label file.

## Classificator data generator
The Class ClassificatorDataGenerator was implemented to create the data set which will be used to train and test the learning algorithm. In the section before the pruned and merged data set was split into ten label files in which each file contains exclusively entries with the same label.
To get a random order of labeled data entries in each data subset for cross-validation a random label, which reflects the label file from which the row should the data entry be read in, was chosen according to the label amount proportions and written into one of the ten data subsets for cross-validation, also randomly chosen.
Each subset for cross-validation should now reflect the label amount proportion of the original data sets and every subset for cross-validation should have approximately the same amount of data entries.

# Quality Measures

The overall task is a classification problem with six distinct classes. To compare our algorithms we use some quality measures which will be described in this chapter.

## Accuracy

The accuracy is a common quality measure for classification problems. It is the number of right classified instances divided by the overall number of instances. This was our main quality measure for the comparison.

## Confusion Matrix

The confusion matrix is used to get a better overview over a single classification result. For N considered classes, the confusion matrix is an NxN matrix. The rows depict the ground truth, the columns the estimated class. Every cell of the matrix „counts“ classifications. For example a 100 in the cell (A,B) means that instances with the real class A, in 100 cases are assigned to the class B by the considered classifier. 
The desired result is a matix in which everything is zero except the values in the main diagonal. This means that all estimated classes are equal to the ground truth, for every instance. 

The confusion matrix helps to get a overview on how the different classes were separated. When two different classes A and B are confused, then the cells (A,B) and (B,A) will have high values. It is a useful visualization of a classification result to see the atony and strength of the corresponding classifier.

We introduce an additional row (N+1) and column (N+1), in which the corresponding values of the row or column are summarized.


# Results for each classifier used

## SVM learning ('rbf' - Kernel)

The SVM learning approach is pretty generic, if 'only' you had the right kernel alsmost every problem may be solved with an svm. In our case it was mostly curiosity of how svms would perform, given some of the default kernels.
The Kernel that outperformed others by far is the 'rbf' kernel, which operates on the distance of the vectors. The kernel is one of the go-to kernels for machine learning problems that often performs well. However we have not searched for a kernel that can exploit specific knowledge.

A problem that came up is that (at least in this implementation) training the SVM and also using it to classify new instances takes a lot of time. By limiting the maximum number of iterations computations can be made in a feasable time, but the SVM does not converge fast enough if to many training samples are provided. Especially as the classification should quite propably run on a mobile phone this is a big disadvantage of the approach.
Using a small number of training samples also means, that 'good' training samples have to be chosen, as biased data can have a large impact on the result. This happend to us and led to accuracy values of 0.9 in training and 0.8 in evaluation sets with the same data distribution, but had an overall accuracy of under 0.3 on randomly selected evaluation data.

Another advantage is that SVMs can not be trained online, but the model needs to be learned in advance.
```
bike       [ 38975  2013    5325    12809   3748   1656    64526  
sit        [ 310    103757  70         43      60       13    104253  
stand      [ 979    344     101185    432     307     168    103415  
walk       [ 11981  1344    5483    82608   12780  6374    120570  
stairsup   [ 7786   610     3874    30364   30097  6939    79670  
stairsdown [ 7121   511     3235    33245   12728  15785   72625  
all        [ 67152  108579  119172  159501  59720  30935   545059  

mean_accuracy:  0.683425751842
```

## Decision Tree learning
When looking at the classes we thought that there are some subclasses that are easily distinguishable by hand. There are some activities that require a lot of motion (cycling, walking, stairs) and others that don't (standing, sitting). Once you know that the activity is likely to be either standing or sitting you could look for the orientation of the phone to tell apart sitting and standing.
This is pretty similar to how a decision tree would work like, so we decided to test how well the decision tree learning from sklearn is able to classify our data.

However the model learned by the decision tree may not be a good generalisation of the underlying association of the attributes. Especially the 'perfect' decision tree that has no error on the training data performs poorly when classifying real data. By generating a tree that aggregates a number of training examples in each node befor splitting it and limiting the minimal number of samples in the leaf nodes overfitting can be counteracted and the accuracy is quite good.

In the implementation of decision trees we use, the model is fitted in advance of classification. However there are also online learning approaches for decision trees. One of those approaches is to save the data items in the leaf node of the tree and to split the nodes as soon as a certain threshold is reached. In contrast to offline learning with this approach it is even harder to find the optimal feature split, as less data items are available to decide on how to split the node.

## Naive Bayes
We used four different machine learning methods and present the results in the following. First, we tried naive bayes. The mean accuracy after running cross validation is: 

mean_accuracy:  0.46783015461

We guess that the accuracy is so low because of the underlying independence assumption. This is not true as they are directly correlated to each other (redundancy of features).
When sitting e.g., the accelerometer won't change it's values ignoring jitter. This also holds for the gyrometer. Another assumption that has been made but must not be true,
is to think that the data is normal distributed. The result for online naive bayes is equal because it does not make a difference when and in which order to read in data.

Below the summarized confusion matrix for the 10 fold crossvalidation is depicted
```
               bike     sit    stand   walk  s-up    s-down     all
bike       [ 174117   55085  128905  210417   51632   22203  642359]
sit        [  25166  692781  283821     210    5132   38331 1045441]
stand      [  24505   74559  896433   24017    5969    3210 1028693]
walk       [  70019  191574  237291  514705  121561   73864 1209014]
stairsup   [  41800   64164  150102  313976  165688   58465  794195]
stairsdown [  50803   77630  115822  275309  101505  103073  724142]
all        [ 386410 1155793 1812374 1338634  451487  299146 5443844]
```
When biking, naive bayes thinks mostly that you walk. Sitting and standing is recognized quite well but there is still some confusion between these two classes. When one goes up or
down the stairs, mostly walking will be recognized again.

## K-Nearest-Neighbors
Looking at k-Nearest-Neighbours, the accuracy improves but it seems not worth to go beyond three neigbours as the accuracy won't improve significantly:

3-NN mean_accuracy:  0.690442637993
5-NN mean_accuracy:  0.699013810978

Most probably, the accuracy increased because of clusters that build. Sitting and standing is well seperated and classified correctly most of the times. The nice seperation is working because they are very different from all other classes as sitting will not change the sensor data (nearly constant) while standing will cause slight movements + jitter. Because walking and biking have more similar vectors they have a lot of false positives/negatives. More drastic is that stairsup and down seem to be very similar to their counterpart and especially walking. Giving this general setting, kNN should only be considered differentiating less similar classes like e.g. sit + stand + walk|bike


               bike     sit    stand   walk    s-up   s-down    all
bike       [ 443651    6098   21499  102999   37575   30536  642358]
sit        [   1378 1040850     792    1547     619     253 1045439]
stand      [   9045    1053 1007032    6758    2692    2112 1028692]
walk       [ 156727    6524   27178  754470  139315  124798 1209012]
stairsup   [ 104552    3895   19061  268989  298120   99577  794194]
stairsdown [  99558    2606   16898  271332  119209  214536  724139]
all        [ 814911 1061026 1092460 1406095  597530  471812 5443834]

# Comparison

Excluding naive bayes, all of the used methods have nearly the same accuracy. Also the classification errors are similar. In general, sitting and standing are well seperable while movement classes (biking, walking, stairsup/stairsdown) share sensor patterns that lead to misclassifications.

Comparing SVM and kNN is also interesting, as the RBF kernel works distance based and is able to recognize clusters. This may explain why SVM and kNN achieve similar results and also why the rbf-kernel seems appropriate to classify the data.

# Future Work

One aspect of the future work would be the improvment of the implemented algorithms. Parameter tuning would be one of those tasks. With SVMs we achieve quite good results, but there is still room for improvement. Due to the limited time we could not test the data with enough different parameters.

Another approach is to use the temporal component. The inertial measurement unit of the telephone gathers a new data item serveral times each second, the activities will not change fast, so accuracy might be improved by combining multiple measurements taken in a short time period. A really simple way of doing this would be to take the majority vote of occuring classifications for the last n measurements.
