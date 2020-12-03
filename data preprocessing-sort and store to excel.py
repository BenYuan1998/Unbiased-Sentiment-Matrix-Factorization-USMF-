#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:58:33 2020

@author: nuoyuan
"""
"""
This Python script sorts the user-item interactions in the subject dataset 
by the dates they took place on and stores the sorted version as an excel file

"""
#import numpy as np
import xlwt
#import pandas as pd
import gzip
from sort import sort_data


def parse(path):
    g=gzip.open(path,"rb")
    for l in g:
        yield eval(l)
def store_data_as_list(path):
    data=list()
    for d in parse(path):
        dictionary=d
        entry=list()
        for key,value in dictionary.items():
            if key=="reviewerID":
                entry.append(value)
            elif key=="asin":
                entry.append(value)
            elif key=="reviewText":
                entry.append(value)
            elif key=="overall":
                entry.append(value)
            elif key=="reviewTime":
                value=value.replace(" ","").replace(",","")
                if len(value)==7:
                    value=[str(value[-4:-1])+str(value[-1]),str(value[0:2]),str(value[2:3])]
                elif len(value)==8:  
                    value=[str(value[-4:-1])+str(value[-1]),str(value[0:2]),str(value[2:4])]
                value="-".join(value)
                entry.append(value)
        data.append(entry)
    data_sorted=sort_data(data)
    return data_sorted
def store_list_into_excel(List,filename):
    workbook=xlwt.Workbook()
    sheet1=workbook.add_sheet("data")
    row=0
    for interaction in List:
        for col,item in enumerate(interaction):
            sheet1.write(row,col,item)
        row+=1
    workbook.save("{}.xls".format(filename))
path="/Users/nuoyuan/Desktop/X-Factory/数据集/reviews_Cell_Phones_and_Accessories_5.json.gz"
data_sorted=store_data_as_list(path)
store_list_into_excel(data_sorted,"Phones_and_Accessories")











