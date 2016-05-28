# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:33:05 2016

@author: Uenal
"""

import os
import numpy as np
import csv
from random import randint

class ClassificatorDataGenerator:
    # the probability of each label and absolute amount of data rows in each label dataset
    #prob = {0.11799731954, 0.19204095488, 0.18896445232, 0.22208828908, 0.14588864045, 0.13302034371}
    probAbs = np.array([642359, 1045441, 1028693, 1209014, 794195, 724142])
    
    #default constructor
    def __init__(self):
        print("init")
        
    # determines and returns next index
    def getNextIndex(self):
        sumProbAbs = np.sum(self.probAbs) # sum up all remaining row amounts
        cumSumProbAbs = np.cumsum(self.probAbs) # calculate the cumulative sum over probAbs
        num = randint(0, sumProbAbs) # get a random number between zero and maximum number of rows
        res = 0
        for a in cumSumProbAbs:
            if num <= a:
                return res
            res += 1
        # if nothing matches return last index
        return len(cumSumProbAbs) - 1
    
    def splitPrunedData(self):
        directory = "../sub_datasets"
        # check if directory for sub_datasets exists and create one if neccessary
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # create reader for all label sets, one for each class label
        readFile = [open("../sperated_data/pruned_data_label_0.csv","r",newline=''),
                     open("../sperated_data/pruned_data_label_1.csv","r",newline=''),
                     open("../sperated_data/pruned_data_label_2.csv","r",newline=''),
                     open("../sperated_data/pruned_data_label_3.csv","r",newline=''),
                     open("../sperated_data/pruned_data_label_4.csv","r",newline=''),
                     open("../sperated_data/pruned_data_label_5.csv","r",newline='')]
        reader = [csv.reader(readFile[0], delimiter=',', quotechar='|'),
                  csv.reader(readFile[1], delimiter=',', quotechar='|'),
                  csv.reader(readFile[2], delimiter=',', quotechar='|'),
                  csv.reader(readFile[3], delimiter=',', quotechar='|'),
                  csv.reader(readFile[4], delimiter=',', quotechar='|'),
                  csv.reader(readFile[5], delimiter=',', quotechar='|')]
        # skip header row in all reader
        for r in reader:
            next(r)
        # put all reader into a reader list
        readerList = []
        readerList.insert(0,reader[0])
        readerList.insert(1,reader[1])
        readerList.insert(2,reader[2])
        readerList.insert(3,reader[3])
        readerList.insert(4,reader[4])
        readerList.insert(5,reader[5])
        
        # create six csv writer, one for each sub_dataset
        fieldnames = ["aX","aY","aZ","gX","gY","gZ","label"]
        writeFile = [open("../sub_datasets/subset_0.csv","w",newline=''),
                     open("../sub_datasets/subset_1.csv","w",newline=''),
                     open("../sub_datasets/subset_2.csv","w",newline=''),
                     open("../sub_datasets/subset_3.csv","w",newline=''),
                     open("../sub_datasets/subset_4.csv","w",newline=''),
                     open("../sub_datasets/subset_5.csv","w",newline=''),
                     open("../sub_datasets/subset_6.csv","w",newline=''),
                     open("../sub_datasets/subset_7.csv","w",newline=''),
                     open("../sub_datasets/subset_8.csv","w",newline=''),
                     open("../sub_datasets/subset_9.csv","w",newline='')]
        writer = [csv.DictWriter(writeFile[0], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[1], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[2], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[3], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[4], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[5], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[6], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[7], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[8], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[9], fieldnames=fieldnames, delimiter=',', quotechar='|')]
        writer[0].writeheader(); writer[1].writeheader(); writer[2].writeheader(); writer[3].writeheader(); writer[4].writeheader();
        writer[5].writeheader(); writer[6].writeheader(); writer[7].writeheader(); writer[8].writeheader(); writer[9].writeheader();
        
        while True:
            try:
                # determine read and write index
                readIn = self.getNextIndex()
                writeIn = randint(0,9)
                row = next(readerList[readIn]) # read in data row from determined read index
                self.probAbs[readIn] -= 1
                
                try:
                    # write in data row from determined write index
                    writer[writeIn].writerow({"aX": row[0], "aY": row[1], "aZ": row[2], "gX": row[3], "gY": row[4], "gZ": row[5], "label": row[6]})
                except ValueError:
                    print("ValueError")
                    continue
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End: "+str(readIn))
                #print(len(readerList))
                #print(self.probAbs)
                readerList.remove(readerList[readIn])
                i=0
                k = []
                for p in self.probAbs:
                    if i != readIn:
                        k.insert(i, p)
                        i += 1
                    elif i == 0:
                        i += 1
                self.probAbs = np.array(k)
                #self.pobAbs = np.delete(self.probAbs, readIn)
                #print(len(readerList))
                #print(self.probAbs)
                if len(readerList) == 0:
                    break
            except:
                print("Exception: "+str(readIn))
                #print(len(readerList))
                #print(self.probAbs)
                #print(row)
                #print(readIn)
                #print(writeIn)
                break
                
        # close all reader and writer files
        for w in writeFile:
            w.close()
        for r in readFile:
            r.close()
        
        print("finished ")