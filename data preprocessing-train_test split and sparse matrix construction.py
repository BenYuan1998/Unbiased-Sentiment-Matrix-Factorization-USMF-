#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:26:07 2020

@author: nuoyuan
"""
"""
This script splits the subject dataset into the training and test sets for rating prediction and top-K recommendation
"""
import xlrd
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Open the subject dataset stored as an xls file
def open_xls(filename,sheetname):
    """
    Parameters
    ----------
    filename: the name of the xls file that stores the subject dataset (remember to affix ".xls" to its end)
    sheetname: the name of the sheet that contains the subject dataset
    
    Returns
    -------
    sheet: the sheet that contains the subject dataset
   """
    workbook=xlrd.open_workbook(filename)
    sheet=workbook.sheet_by_name(sheetname)
    return sheet

# Remove all user-item interactions with missing entries and store the resulting dataset as a dataframe
def data_cleaning(dataset):
    """
    Parameters
    ----------
    dataset: the sheet that contains the subject dataset
    
    Returns
    -------
    df: the subset of the subject dataset containing userids, itemids, reviews, ratings, and dates stored as a dataframe
    """
    data={"userid":dataset.col_values(0),
          "itemid":dataset.col_values(1),
          "review":dataset.col_values(2),
          "rating":dataset.col_values(3),
          "date":dataset.col_values(4)
          }
    pd.set_option("max_colwidth",max([len(review) for review in data["review"]]))
    df=pd.DataFrame.from_dict(data,orient="columns")
    df.dropna(axis=0,how="any",inplace=True)
    return df 

# Assign indices to unique userids and itemids respectively
def id_index_mapping(IDs):
    """
    Parameters
    -----------
   indices: userids/itemids stored as a numpy array
   
   Returns
   -------
   id_to_index: the one-to-one mappings from userids/itemids to their corresponding indices stored as a dictionary
   """
    id_to_index=dict()
    for index,ID in enumerate(np.unique(IDs).tolist()):
        id_to_index[ID]=index
    return id_to_index

# Replace userids and itemids with their corresponding indices
def replace_id_by_index(df,userid_to_index,itemid_to_index):
    """
    Parameters
    ----------
    df: the subset of the subject dataset containing userids, itemids, reviews, ratings, and dates stored as a dataframe
    userid_to_index: the one-to-one mappings from userids to their corresponding indices stored as a dictionary
    itemid_to_index: the one-to-one mappings from itemids to their corresponding indices stored as a dictionary
        
    Returns
    -------
    df_new: df modified by replacing userids and itemids by their respective corresponding indices
    """
    userids=df["userid"]
    itemids=df["itemid"]
    def map_ids(column,mapper):
        return mapper[column]
    userindices=userids.apply(map_ids,args=[userid_to_index])
    itemindices=itemids.apply(map_ids,args=[itemid_to_index])
    df["userid"]=userindices
    df["itemid"]=itemindices
    df.rename(columns={"userid":"user_index","itemid":"item_index"},inplace=True)
    df_new=df
    return df_new


#n_users=len(userid_to_index)
#n_items=len(itemid_to_index)
#print(n_items)
#print(df_new.shape[0])
#print("The sparsity of the dataset is: {}%".format((df_new.shape[0]/(n_users*n_items))*100))
#n_reviews_per_user=dict()
#users=df_new["user_index"].values.tolist()
#print(len(users))
#for user in users:
    #n_reviews_per_user[user]=n_reviews_per_user.get(user,0)+1

# Extract from the subject dataset a denser subset containing only users who made at least 2K reviews for top-K recommendation and name it subset2K
def subset_extraction(df_new,k):
    """
    Parameters
    ----------
    df_new : the subject dataset stored as a dataframe
    k : the number of items per user to be used for assessing model performance on the ranking task 

    Returns
    -------
    df_subset: the subset of df_new exclusively containing users with at least 2K ratings/reviews stored as a dataframe
    """
    unique_userindices=np.unique(df_new["user_index"].values).tolist()
    df_subset_indices=list()
    for useri in unique_userindices:
        useri_indices=df_new[df_new["user_index"]==useri].index.tolist()
        if len(useri_indices)>=2*k:
            df_subset_indices+=useri_indices
    df_subset=df_new.loc[df_subset_indices]
    return df_subset
          
# Split the subject dataset into the training and test sets 
def training_set(df_new,n):
    """
    Parameters
    ----------
    df_new: the subject dataset stored as a dataframe
    n: the number of most recently reviewed items for each user

    Returns
    -------
    df_training: the training set stored as a dataframe
    """
    unique_userindices=np.unique(df_new["user_index"].values).tolist()
    training_indices=list()
    for useri in unique_userindices:
        useri_indices=df_new[df_new["user_index"]==useri].index.tolist()
        for i in range(len(useri_indices)-n):
            training_indices.append(useri_indices[i])
    df_training=df_new.loc[training_indices]
    return df_training     

def test_set(df_new,training_indices):
    """
    Parameters
    ----------
    df_new: the subject dataset stored as a dataframe
    training_indices: the indices corresponding to the user-item interactions used for training

    Returns
    -------
    df_test: the test set stored as a dataframe
    """
    df_test=df_new[~df_new.index.isin(training_indices)]
    return df_test


filename="Amazon_digitalmusic.xls"
sheet="data"
#dataset=open_xls(filename,sheet)
#df=data_cleaning(dataset)
#userids=df["userid"].values
#itemids=df["itemid"].values
#userid_to_index=id_index_mapping(userids)
#itemid_to_index=id_index_mapping(itemids)
#df_new=replace_id_by_index(df,userid_to_index,itemid_to_index)

#df_subset=subset_extraction(df_new,5)
#df_training=training_set(df_subset,2)
#training_indices=df_training.index.tolist()
#df_test=test_set(df_subset,training_indices)





