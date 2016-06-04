'''
Created on 03.06.2016

@author: freddy
'''

import numpy

'''
# example of how to use the eval class:
        
groundtruth      = numpy.array([0,1,2,3,4,5,3,4,3,2,3,4,2,3,4,1,2,3,4,5,3,4,3,2,3,4,2,3,2,4])
estimated_values = numpy.array([0,1,2,3,4,5,3,4,3,2,3,4,2,3,4,3,2,2,3,4,5,3,4,3,2,3,4,2,3,4])

# create eval instance
test_eval = eval(groundtruth, estimated_values)

#print all generated data
test_eval.print_eval()

'''

class eval:
    
    numberOfInstances = -1
    numberOfClasses   = 6
    
    groundtruth       = numpy.array(int)
    estimated_values  = numpy.array(int)
    
    conf_matrix       = numpy.zeros((numberOfClasses + 1, numberOfClasses + 1), int)
    
    overallNmberOfTrueClassifiedObjects  = 0
    overallNmberOfFalseClassifiedObjects = 0
    
    overall_tp = -1
    overall_tn = -1
    overall_fp = -1
    overall_fn = -1
    
    def __init__(self, _groundtruth, _estimatedValues):
        self.groundtruth = _groundtruth
        self.estimated_values = _estimatedValues
        
        self.numberOfInstances = len(self.groundtruth)
        
        self.create_confusion_matrix()
        self.create_tp_tn_fp_fn_from_confusion_matrix()
        
        
    def create_confusion_matrix(self):
        
        for i in range(self.numberOfInstances):
            
            index_groundtruth = self.groundtruth[i]
            index_found_class = self.estimated_values[i]
            
            test = self.conf_matrix[index_groundtruth][index_found_class]       + 1
            
            self.conf_matrix[index_groundtruth][index_found_class]       = test
            self.conf_matrix[index_groundtruth][self.numberOfClasses]    = self.conf_matrix[index_groundtruth][self.numberOfClasses]    + 1
            self.conf_matrix[self.numberOfClasses][index_found_class]    = self.conf_matrix[self.numberOfClasses][index_found_class]    + 1
            self.conf_matrix[self.numberOfClasses][self.numberOfClasses] = self.conf_matrix[self.numberOfClasses][self.numberOfClasses] + 1

            if(index_groundtruth == index_found_class):
                self.overallNmberOfTrueClassifiedObjects += 1
            else:
                self.overallNmberOfFalseClassifiedObjects += 1
            
    def create_tp_tn_fp_fn_from_confusion_matrix(self):
                
        self.overall_tp = 0
        self.overall_tn = 0
        self.overall_fp = 0
        self.overall_fn = 0
        
        # hier noch die wahren werte abziehen
        
        for i in range(self.numberOfClasses):
            self.overall_tp += self.conf_matrix[i][i]
            self.overall_tn += self.conf_matrix[i][i]
        
        for i in range(self.numberOfClasses):
            self.overall_fp += self.conf_matrix[self.numberOfClasses][i] - self.conf_matrix[i][i]

        for i in range(self.numberOfClasses):
            self.overall_fn += self.conf_matrix[i][self.numberOfClasses] - self.conf_matrix[i][i]

    def print_eval(self):
        
        self._print_general_values()
        self._print_four_values()
        self._print_quality_values(self.overall_tp, self.overall_tn, self.overall_fp, self.overall_fn)
        self._print_conf_matrix()
        self._print_conf_matrix_probabilities()

    def _print_general_values(self):
        print('general values:\n')
        
        print('number of Classes:             ' + str(self.numberOfClasses))
        print('number of Instances:           ' + str(self.numberOfInstances)) 
        print('overall true classifications:  ' + str(self.overallNmberOfTrueClassifiedObjects))
        print('overall false classifications: ' + str(self.overallNmberOfFalseClassifiedObjects))
        
        print('\n')
        
    def _print_conf_matrix(self):
        print('overall confusion matrix:\n')
        
        print('gt\est\test_0\test_1\test_2\test_3\test_4\test_5\test_all')
        
        for row in range(self.numberOfClasses + 1):
            if(row == 6):
                line = 'gt_all\t'
            else:
                line = 'gt_' + str(row) + '\t'
                
            for col in range(self.numberOfClasses + 1):
                line += str(self.conf_matrix[row][col]) + '\t'
            
            print(line)  
            
        print('\n')  
        
    def _print_conf_matrix_probabilities(self):
        print('overall confusion matrix with probability values:\n')
        
        print('gt\est\test_0\test_1\test_2\test_3\test_4\test_5\test_all')
        
        for row in range(self.numberOfClasses + 1):
            if(row == 6):
                line = 'gt_all\t'
            else:
                line = 'gt_' + str(row) + '\t'
                
            for col in range(self.numberOfClasses + 1):
                line += str("%.3f" % (self.conf_matrix[row][col] / self.numberOfInstances)) + '\t'
            
            print(line)  
            
        print('\n')  
        
    def _print_quality_values(self, tp, tn, fp, fn):
        print('quality values:\n')
        
        sensitivity = tp / (tp + fn)
        specificy   = tn / (tn + fp)
        
        accuracy    = (tp + tn) / (tp + fp + fn + tn)
        f1_score    = (2*tp) / (2*tp + fp + fn) 
        
        print('sensitivity: ' + str(sensitivity))
        print('specificy:   ' + str(specificy))
        print('accuracy:    ' + str(accuracy))
        print('f1_score:    ' + str(f1_score))
        
        print('\n')
        
    def _print_four_values(self):
        print('additional values:\n')
        
        print('true positives:  ' + str(self.overall_tp))   
        print('true negative:   ' + str(self.overall_tn))
        print('false positives: ' + str(self.overall_fp))
        print('false negatives: ' + str(self.overall_fn))
        
        print('\n')
