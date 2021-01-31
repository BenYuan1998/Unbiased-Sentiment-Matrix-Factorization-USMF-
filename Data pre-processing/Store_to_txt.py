#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:58:33 2020

@author: nuoyuan
"""
"""
This Python script sorts the user-item interactions in the subject dataset 
by the dates they took place on and stores the sorted version as a txt file.

"""

import os
import json
from sort import sort_data
import numpy as np

def parse(path):
    g=open(path)
    for l in g:
        yield json.loads(l)
def store_data_as_list(path):
    data=list()
    for d in parse(path):
        dictionary=d
        entry=list()
        keys=dictionary.keys()
        columns=["reviewerID","asin","reviewText","overall","reviewTime"]
        for column in columns:
            for key in keys:
                if column==key:
                    value=dictionary[key]
                    if key=="reviewTime":
                        value=value.replace(" ","").replace(",","")
                        if len(value)==7:
                            value=[str(value[-4:-1])+str(value[-1]),str(value[0:2]),str(value[2:3])]
                        elif len(value)==8:
                            value=[str(value[-4:-1])+str(value[-1]),str(value[0:2]),str(value[2:4])]
                        else:
                            print("The above two categories for classifying date data are not exhaustive.")
                            break
                        value="-".join(value)
                            
                    entry.append(value)
        data.append(entry)
    data_sorted=sort_data(data)
    #data_sorted=np.array(sort_data(data))
    #data_sorted=np.unique(data_sorted).tolist() # remove any duplicates  
    return data_sorted
def store_list_into_txt(List,file_path,filename):
    name_of_file=os.path.join(file_path,filename+".txt")
    with open(name_of_file,"w",encoding="utf-8") as f:
        for interaction in List:
            f.write(str(interaction)+"\n")
            

        










