# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:33:05 2016

@author: Uenal
"""

import os
import numpy as np
import csv

class PrunedDataSeparator:
    
    #default constructor
    def __init__(self):
        print("init")
    
    def spitPrunedData(self):
        directory = "../sperated_data"
        # check if directory exists and create one if neccessary
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # data path for pruned data, open file and create reader for csv file
        pruned_data_path = "../pruned_data.csv"
        file = open(pruned_data_path, "r")
        reader = csv.reader(file, delimiter=',', quotechar='|')
        next(reader)
        
        # create six csv writer, one for each class label
        fieldnames = ["aX","aY","aZ","gX","gY","gZ","label"]
        writeFile = [open("../sperated_data/pruned_data_label_0.csv","w",newline=''),
                     open("../sperated_data/pruned_data_label_1.csv","w",newline=''),
                     open("../sperated_data/pruned_data_label_2.csv","w",newline=''),
                     open("../sperated_data/pruned_data_label_3.csv","w",newline=''),
                     open("../sperated_data/pruned_data_label_4.csv","w",newline=''),
                     open("../sperated_data/pruned_data_label_5.csv","w",newline='')]
        writer = [csv.DictWriter(writeFile[0], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[1], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[2], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[3], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[4], fieldnames=fieldnames, delimiter=',', quotechar='|'),
                  csv.DictWriter(writeFile[5], fieldnames=fieldnames, delimiter=',', quotechar='|')]
        writer[0].writeheader(); writer[1].writeheader(); writer[2].writeheader();
        writer[3].writeheader(); writer[4].writeheader(); writer[5].writeheader();
        
        while True:
            try:
                row = next(reader)
                try:
                    label = int(row[6])
                    writer[label].writerow({"aX": row[0], "aY": row[1], "aZ": row[2], "gX": row[3], "gY": row[4], "gZ": row[5], "label": label})
                except ValueError:
                    print("ValueError")
                    continue
                
            except csv.Error:
                print("error")
            except StopIteration:
                print("Iteration End")
                break
        
        for w in writeFile:
            w.close()
        
        print("finished ")