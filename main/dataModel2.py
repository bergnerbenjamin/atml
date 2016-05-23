'''
Created on 23.05.2016

@author: uenal
'''

import numpy
import csv

#a class which represents the data model
class dataModel2:
    
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
    class_mapping_dict = {'bike' : 'a', 'sit' : 'b', 'stand' : 'c', 'walk' : 'd', 'stairsup' : 'e', 'stairsdown' : 'f', 'null' : 'g'}
    
    # the device name mapping to numbers
    device_mapping_dict = {'nexus4_1' : 'a', 'nexus4_2' : 'b', 's3_1' : 'c', 's3_2' : 'd', 's3mini_1' : 'e', 's3mini_2' : 'f', 'samsungold_1' : 'g', 'samsungold_2' : 'h'}
    
    # stores the class labels of the corresponding instances in the feature vector
    class_label_array = numpy.array([int])
    
    #default constructor, creates the feature vector
    def __init__(self):
        print("init")
        self.data = []
        self.check_data()
        #self.data_pruning()

    def check_data(self):
        # data paths for accelerator and gyroscope data, open file and create reader for both csv files
        accelerator_data_path = "../Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv"
        gyroskope_data_path   = "../Activity recognition exp/Activity recognition exp/Phones_gyroscope.csv"
        accFile = open(accelerator_data_path, "r")
        gyrFile = open(gyroskope_data_path,"r")
        rAcc = csv.reader(accFile, delimiter=',', quotechar='|')
        rGyr = csv.reader(gyrFile, delimiter=',', quotechar='|')
        next(rAcc)
        next(rGyr)
        
        # go through all accelerator data
        lineNumber = 0
        
        print("start parsing accelerator data")
        minX = minY = minZ = 99999.99999
        maxX = maxY = maxZ = -99999.99999
        # parse data parallel line by line
        while True:
            try:
                accRow = next(rAcc)
                try:
                    x = float(accRow[3]); y = float(accRow[4]); z = float(accRow[5])
                except ValueError:
                    continue
                
                minX = min(minX, x)
                minY = min(minY, y)
                minZ = min(minZ, z)
                
                maxX = max(maxX, x)
                maxY = max(maxY, y)
                maxZ = max(maxZ, z)
                
                lineNumber += 1
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End")
                break
        
        accFile.close()
        
        print("accMinX: "+str(minX))
        print("accMaxX: "+str(maxX))
        print("accMinY: "+str(minY))
        print("accMaxY: "+str(maxY))
        print("accMinZ: "+str(minZ))
        print("accMaxZ: "+str(maxZ))
        print("number of lines(acc): "+str(lineNumber))
        print("end parsing accelerator data")
        
        # go through all gyroscope data
        lineNumber = 0
        minX = minY = minZ = 99999.99999
        maxX = maxY = maxZ = -99999.99999
        print("start parsing gyroscope data")
        # parse data parallel line by line
        while True:
            try:
                gyrRow = next(rGyr)
                try:
                    x = float(gyrRow[3]); y = float(gyrRow[4]); z = float(gyrRow[5])
                except ValueError:
                    continue
                
                minX = min(minX, x)
                minY = min(minY, y)
                minZ = min(minZ, z)
                
                maxX = max(maxX, x)
                maxY = max(maxY, y)
                maxZ = max(maxZ, z)
                
                lineNumber += 1
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End")
                break
        
        gyrFile.close()
        
        print("gyrMinX: "+str(minX))
        print("gyrMaxX: "+str(maxX))
        print("gyrMinY: "+str(minY))
        print("gyrMaxY: "+str(maxY))
        print("gyrMinZ: "+str(minZ))
        print("gyrMaxZ: "+str(maxZ))
        print("number of lines(gyr): "+str(lineNumber))
        print("end parsing gyroscope data")
        
    def data_pruning(self):
        # data paths for accelerator and gyroscope data
        accelerator_data_path = "../Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv"
        gyroskope_data_path   = "../Activity recognition exp/Activity recognition exp/Phones_gyroscope.csv"
        pruned_data_path = "../pruned_data.csv"
        
        accFile = open(accelerator_data_path, "r")
        gyrFile = open(gyroskope_data_path,"r")
        # create reader for both data sets and skip first row
        rAcc = csv.reader(accFile, delimiter=',', quotechar='|')
        rGyr = csv.reader(gyrFile, delimiter=',', quotechar='|')
        next(rAcc)
        next(rGyr)
        
        # create writer for writing new, merged and pruned data
        fieldnames = ["aX","aY","aZ","gX","gY","gZ","user","device","label"]
        writeFile = open(pruned_data_path,"w",newline='')
        writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',', quotechar='|')
        writer.writeheader()
        
        lineNumber = 0
        
        print("start parsing and pruning")
        # parse data parallel line by line, remove invalid and faulty data, merge the data
        while True:
            try:
                accRow, gyrRow = self.getNextRows(rAcc, rGyr)
                
                lineNumber += 1
                writer.writerow({"aX": accRow[3], "aY": accRow[4], "aZ": accRow[5], "gX": gyrRow[3], "gY": gyrRow[4], "gZ": gyrRow[5], "user": accRow[6], "device": self.device_mapping_dict[accRow[8]], "label": self.class_mapping_dict[accRow[9]]})
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End")
                break
        
        accFile.close()
        gyrFile.close()
        writeFile.close()
        
        print("number of lines: "+str(lineNumber))
        print("end parsing and pruning")
            
    # check if row data are valid
    def getNextRows(self, rAcc, rGyr):
        accRow = next(rAcc)
        gyrRow = next(rGyr)
        return (accRow, gyrRow)
            