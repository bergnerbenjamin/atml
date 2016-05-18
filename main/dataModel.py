'''
Created on 11.05.2016

@author: freddy
'''

import numpy
import csv

#a class which represents the data model
class dataModel:
    
    #feature vector, every row represents an instance
    #the colums are as follows:
    #0: accelerator_x
    #1: accelerator_y
    #2: accelerator_z
    #3: gyroscope_x
    #4: gyroscope_y
    #5: gyroscope_z
    feature_vector = numpy.array([float, float, float, float, float, float])
    
    #the six classes are mapped to numbers
    class_mapping_dict = {'bike' : 0, 'sit' : 1, 'stand' : 2, 'walk' : 3, 'stairsup' : 4, 'stairsdown' : 5, 'null' : 6}
    
    # stores the class labels of the corresponding instances in the feature vector
    class_label_array = numpy.array([int])
    
    #default constructor, creates the feature vector
    def __init__(self):
        self.data = []
        self.create_feature_vector()

    #create the feature vector by combining the accelerator and gyroscope data
    def create_feature_vector(self):
        accelerator_data = self.read_in_data("../Activity recognition exp/test2long.csv")
        gyroskope_data   = self.read_in_data("../Activity recognition exp/test3long.csv")
        
        print('start reading')
        #accelerator_data = self.read_in_data("../Activity recognition exp/Phones_accelerometer.csv")
        print('first finished')
        #gyroskope_data = self.read_in_data("../Activity recognition exp/Phones_gyroscope.csv")
        print('end reading')
        
        size = -1
        
        # handle differences in length
        if len(accelerator_data) <= len(gyroskope_data):
            size = len(accelerator_data)
        else:
            size = len(gyroskope_data)
            
        print('-----------')
        print(accelerator_data)
        print('-----------')
        print(gyroskope_data)
        print('-----------')
        
        #when the idices of both vectors are not equal in one line, delete the smaller
        #one. When the smaller one is 0, delete the higher one.
        for line_number in range(0,size-1):
            
            #security check, because of deletion of lines
            if line_number >= len(accelerator_data) or line_number >= len(gyroskope_data):
                break
            
            index_acc = int(accelerator_data[line_number][0])
            index_gyr = int(gyroskope_data[line_number][0])
            
            print('line: ' + str(line_number) + 'index_acc: ' + str(index_acc) + ' index_gyr: ' + str(index_gyr))
            
            if not isinstance(index_acc, int):
                print('acceleator index no int, line: ' + str(int(index_acc)) + ' '+ str(line_number))
            if not isinstance(index_gyr, int):
                print('gyroscope index no int, line: ' +  str(int(index_gyr)) + ' '+str(line_number))
            
            #error in indices
            if index_acc != index_gyr:
                
                if index_acc < index_gyr:
                    if index_acc != 0:
                        accelerator_data = numpy.delete(accelerator_data, line_number, 0)
                        print('accelerator line deleted: ' + str(line_number))
                    else:
                        gyroskope_data = numpy.delete(gyroskope_data, line_number, 0)
                        print('gyroscope line deleted: ' + str(line_number))
                        
                else:
                    if index_gyr != 0:
                        gyroskope_data = numpy.delete(gyroskope_data, line_number, 0)
                        print('gyroscope line deleted: ' + str(line_number))
                    else:
                        accelerator_data = numpy.delete(accelerator_data, line_number, 0)
                        print('accelerator line deleted: ' + str(line_number))
                
                #decremet line_numer so that the same line will be checked in the next iteration
                line_number -= line_number
                continue
            
            # class labels are not the same -> do not consider the line
            if int(accelerator_data[line_number][4]) != int(gyroskope_data[line_number][4]):
                accelerator_data = numpy.delete(accelerator_data, line_number, 0)
                gyroskope_data = numpy.delete(gyroskope_data, line_number, 0)
                
                #decremet line_numer so that the same line will be checked in the next iteration
                line_number -= line_number
                continue
                
            # if both class labels are null, also delete the line
            if int(accelerator_data[line_number][4]) == 6 or int(gyroskope_data[line_number][4]) == 6:                                                     
                accelerator_data = numpy.delete(accelerator_data, line_number, 0)
                gyroskope_data = numpy.delete(gyroskope_data, line_number, 0)
                
                #decremet line_numer so that the same line will be checked in the next iteration
                line_number -= line_number
                continue
            
            #set the label in the class label array
            self.class_label_array = numpy.append(self.class_label_array, int(accelerator_data[line_number][4]))
            
        print(len(accelerator_data))
        print(len(gyroskope_data))
        print(len(accelerator_data[0]))
        print(len(gyroskope_data[0]))
        print('-----------')
        print(accelerator_data)
        print('-----------')
        print(gyroskope_data)
        print('-----------')
        
        # handle differences in length
        if len(accelerator_data) <= len(gyroskope_data):
            feature_vector_size = len(accelerator_data)
        else:
            feature_vector_size = len(gyroskope_data)
            
        feature_vector_temp = numpy.empty((feature_vector_size, 6), dtype=float)

        #join the two feature vectors to one
        # maybe better alternative: self.feature_vector = numpy.hstack((accelerator_data, gyroskope_data))
        for line_number in range(feature_vector_size):
            feature_vector_temp[line_number][0] = accelerator_data[line_number][1] 
            feature_vector_temp[line_number][1] = accelerator_data[line_number][2]
            feature_vector_temp[line_number][2] = accelerator_data[line_number][3]
            feature_vector_temp[line_number][3] = gyroskope_data[line_number][1] 
            feature_vector_temp[line_number][4] = gyroskope_data[line_number][2]
            feature_vector_temp[line_number][5] = gyroskope_data[line_number][3]
        
        print('feature_vector_temp after assignment')
        print(feature_vector_temp)
        
        print('class labels:')
        print(self.class_label_array)
        
        self.feature_vector = feature_vector_temp

        
    #read the data from the given path and return the result, a five dimensional vector
    #0 index
    #1 data_x (accelerator oder gyroscope)
    #2 data_y (accelerator oder gyroscope)
    #3 data_z (accelerator oder gyroscope)
    #4 class as int
    def read_in_data(self,rel_path):#rel_path = "../Activity recognition exp/test2.csv"
        with open(rel_path, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            
            #skip first line with column descriptions
            next(csv_reader)
            
            data_vector = numpy.array([[int(row[0]),float(row[3]),float(row[4]),float(row[5]),self.class_mapping_dict[row[9]]] for row in csv_reader])
            
            return data_vector

        
            