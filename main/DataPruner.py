'''
Created on 23.05.2016

@author: uenal
'''

import numpy as np
import csv

# compare function like in Java the compareTo method
# returns -1, if a < b
#          0, if a == b
#         +1, if a > b
def compare(a,b):
    return ((a > b) - (a < b))

#a class which represents the data model
class DataPruner:
    
    #the six classes are mapped to numbers
    class_mapping_dict = {'bike' : 0, 'sit' : 1, 'stand' : 2, 'walk' : 3, 'stairsup' : 4, 'stairsdown' : 5, 'null' : 6}
    
    # the device name mapping to numbers
    #device_mapping_dict = {'nexus4_1' : 0, 'nexus4_2' : 1, 's3_1' : 2, 's3_2' : 3, 's3mini_1' : 4, 's3mini_2' : 5, 'samsungold_1' : 6, 'samsungold_2' : 7}
    
    #default constructor
    def __init__(self):
        print("init")
        
    def countPrunedLabelClasses(self):
        # data path for pruned data, open file and create reader for csv file
        pruned_data_path = "../pruned_data.csv"
        file = open(pruned_data_path, "r")
        reader = csv.reader(file, delimiter=',', quotechar='|')
        next(reader)
        
        print("start parsing accelerator data")
        count = np.int_(np.zeros(7))
        # parse data line by line
        while True:
            try:
                row = next(reader)
                try:
                    count[row[6]] += 1
                except ValueError:
                    continue
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End")
                break
        
        file.close()
        print("end parsing accelerator data")
        
        #print result
        print("Result:")
        print(self.class_mapping_dict)
        print("Pruned Data:")
        print(count)
        
    def countLabelClasses(self):
        # data paths for accelerator and gyroscope data, open file and create reader for both csv files
        accelerator_data_path = "../Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv"
        gyroskope_data_path   = "../Activity recognition exp/Activity recognition exp/Phones_gyroscope.csv"
        accFile = open(accelerator_data_path, "r")
        gyrFile = open(gyroskope_data_path,"r")
        rAcc = csv.reader(accFile, delimiter=',', quotechar='|')
        rGyr = csv.reader(gyrFile, delimiter=',', quotechar='|')
        next(rAcc)
        next(rGyr)
        
        print("start parsing accelerator data")
        accCount = np.int_(np.zeros(7))
        gyrCount = np.int_(np.zeros(7))
        # parse acc data line by line
        while True:
            try:
                accRow = next(rAcc)
                try:
                    accCount[self.class_mapping_dict[accRow[9]]] += 1
                except ValueError:
                    continue
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End")
                break
        
        accFile.close()
        print("end parsing accelerator data")
        print("start parsing gyroscope data")
        # parse gyr data line by line
        while True:
            try:
                gyrRow = next(rGyr)
                try:
                    gyrCount[self.class_mapping_dict[gyrRow[9]]] += 1
                except ValueError:
                    continue
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End")
                break
        
        gyrFile.close()
        print("end parsing gyroscope data")
        
        #print result
        print("Result:")
        print(self.class_mapping_dict)
        print("Acc:")
        print(accCount)
        print("Gyr:")
        print(gyrCount)

    def dataPruning(self):
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
        fieldnames = ["aX","aY","aZ","gX","gY","gZ","label"]
        writeFile = open(pruned_data_path,"w",newline='')
        writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',', quotechar='|')
        writer.writeheader()
        
        lineNumber = 0
        
        print("start parsing and pruning")
        # parse data parallel line by line, remove invalid and faulty data, merge the data
        while True:
            try:
                # get next matching acc and gyr rows
                accRow, gyrRow = self.getNextMatchingRows(rAcc, rGyr)
                
                # try to parse the needed x,y,z values as float from the datasets
                try:
                    aX = float(accRow[3]); aY = float(accRow[4]); aZ = float(accRow[5])
                    gX = float(gyrRow[3]); gY = float(gyrRow[4]); gZ = float(gyrRow[5])
                except ValueError:
                    continue
                
                lineNumber += 1
                # write parsed valid data into pruned data set
                writer.writerow({"aX": aX, "aY": aY, "aZ": aZ, "gX": gX, "gY": gY, "gZ": gZ, "label": self.class_mapping_dict[accRow[9]]})
                #print(lineNumber)
                
            except csv.Error:
                print("csv error")
            except StopIteration:
                print("Iteration End")
                break
        
        # close files
        accFile.close()
        gyrFile.close()
        writeFile.close()
        
        print("number of lines: "+str(lineNumber))
        print("end parsing and pruning")
            
    # check if row data are valid and find next data rows with same index value
    def getNextMatchingRows(self, rAcc, rGyr, loadInstantNextRows = True):
        # get next rows
        if loadInstantNextRows:
            accRow = next(rAcc)
            gyrRow = next(rGyr)
        
        while True:
            # skip invalid rows until a valid is found
            while not self.isRowValid(accRow):
                accRow = next(rAcc)
            while not self.isRowValid(gyrRow):
                gyrRow = next(rGyr)
            
            # get index of rows
            aIndex = accRow[0]; gIndex = gyrRow[0]
            cmp = compare(aIndex, gIndex)
            
            # matching rows with same index number and label found, return rows
            if cmp == 0:
                # matching indicies found -> return the rows
                return (accRow, gyrRow)
            if cmp < 0:
                # index of acc is lower than gyr
                if aIndex == 0:
                    # special case if index is zero => skip gyr rows till the index is also zero
                    self.skipReaderTillNextStartingIndex(rGyr)
                else:
                    # load next acc row
                    accRow = next(rAcc)
                # repeat recursively
                continue
            else:
                # index of gyr is lower than acc
                if gIndex == 0:
                    # special case if index is zero => skip acc rows till the index is also zero
                    self.skipReaderTillNextStartingIndex(rAcc)
                else:
                    # load next gyr row and repeat recursively
                    gyrRow = next(rGyr)
                # repeat recursively
                continue
                
    
    # find and skip reader till row have index 0
    def skipReaderTillNextStartingIndex(self, reader, row):
        index = reader[0]
        while index != 0:
            index = next(reader)[0]
        
        return row
        
    # check if index is int, x,y,z if float and label not 'null'
    def isRowValid(self, row):
        try:
            int(row[0]); float(row[3]); float(row[4]); float(row[5]); float(row[3]);
        except:
            return False
        
        return (not row[9] == 'null');