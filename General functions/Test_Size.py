# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:22:55 2020

@author: Administrator
"""
import os
import sys
import numpy as np
import pandas as pd


path_train_test_split=r"../Data pre-processing"
os.chdir(path_train_test_split)

import Train_Test_Split

def test_size(df,N,m):
    """
    Parameters
    ----------
    df: The original dataset stored as a pandas dataframe.
    N : The threshold to extract N-core subset from the original subject dataset
    m : The number of most recently reviewed items for each user to be included in the test set

    Returns
    -------
    test_percentage: the proportion of the N-core subset used for testing
    """
    df_ncore=Train_Test_Split.Ncore(df,N)
    df_train=Train_Test_Split.training_set(df_ncore,m)
    training_indices=df_train.index.tolist()
    df_test=Train_Test_Split.test_set(df_ncore,training_indices)
    test_percentage=df_test.shape[0]/df_ncore.shape[0]
    return test_percentage

path_preprocessing=r"C:\Users\Administrator\Desktop\上科大\代码\Data pre-processing"
sys.path.append(path_preprocessing)
import Train_Test_Split
path_dataset=r"C:\Users\Administrator\Desktop\上科大\数据集\Yelp_dataset\Phoenix_City\Food_Restaurant_Reviews_Phoenix_City_Sorted.txt"
df=Train_Test_Split.txt_to_dataframe(path_dataset)
userids=df["userid"].values
itemids=df["itemid"].values
userid_to_index=Train_Test_Split.id_index_mapping(userids)
itemid_to_index=Train_Test_Split.id_index_mapping(itemids)
df=Train_Test_Split.replace_id_by_index(df=df, userid_to_index=userid_to_index, itemid_to_index=itemid_to_index)
N=10
m=3
print(test_size(df=df,N=N,m=m))






