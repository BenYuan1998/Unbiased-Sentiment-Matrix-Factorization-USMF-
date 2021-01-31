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
import os


# Save the subject dataset originally stored as a txt file into a Pandas dataframe
def txt_to_dataframe(dataset_path):
    """
    Parameters
    ----------
    dataset_path: The absolute path of the subject dataset.

    Returns
    -------
    df: The subject dataset stored as a Pandas dataframe.
    """
    data=dict()
    user_list=[]
    item_list=[]
    review_list=[]
    rating_list=[]
    date_list=[]
    with open(dataset_path,"r") as f:
        for interaction in f.readlines():
            interaction=eval(interaction.strip("\n"))
            if len(interaction)<5:continue
            user=interaction[0]
            item=interaction[1]
            review=interaction[2]
            rating=interaction[3]
            date=interaction[4]
            user_list.append(user)
            item_list.append(item)
            review_list.append(review)
            rating_list.append(rating)
            date_list.append(date)
    data["userid"]=user_list
    data["itemid"]=item_list
    data["review"]=review_list
    data["rating"]=rating_list
    data["date"]=date_list
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
    if "userid" in list(df.columns):
        userids=df["userid"]
        itemids=df["itemid"]
    else:
        userids=df["user_index"]
        itemids=df["item_index"]
    def map_ids(column,mapper):
        return mapper[column]
    userindices=userids.apply(map_ids,args=[userid_to_index])
    itemindices=itemids.apply(map_ids,args=[itemid_to_index])
    df["userid"]=userindices
    df["itemid"]=itemindices
    df.rename(columns={"userid":"user_index","itemid":"item_index"},inplace=True)
    df_new=df
    return df_new



# Extract from the subject dataset a denser subset containing only users who made at least N reviews.
def Ncore(df_new,N):
    """
    Parameters
    ----------
    df_new : the subject dataset stored as a dataframe
    N : the number of reviews contained in the test set for each user

    Returns
    -------
    df_subset: the subset of df_new exclusively containing users with at least 2K ratings/reviews stored as a dataframe
    """
    df_subset_indices=list()
    unique_userindices=np.unique(df_new["userid"].values).tolist()
    for useri in unique_userindices:
        print(useri)
        useri_indices=df_new[df_new["userid"]==useri].index.tolist()
        if len(useri_indices)>=N:
            df_subset_indices+=useri_indices
        df_ncore=df_new.loc[df_subset_indices]   
    return df_ncore
# Split the subject dataset into the training and test sets 


def training_set(df_new,m):
    """
    Parameters
    ----------
    df_new: the subject dataset stored as a dataframe
    m: the number of most recent interactions for each user to be included in the test/validation set.
    Returns
    -------
    df_train: the training set stored as a dataframe
   """
    unique_users_indices_original=np.unique(df_new["user_index"].values).tolist()
    unique_items_indices_original=np.unique(df_new["item_index"].values).tolist()
    training_indices=list()
    for user in unique_users_indices_original:
        interacted_items_indices=df_new[df_new["user_index"]==user].index.tolist()
        training_indices.append(interacted_items_indices[0]) # For user in the original dataset, select his/her most early interaction to the training set.
    unique_items_indices_training=np.unique(df_new.loc[training_indices]["item_index"].values).tolist()
    for item_original in unique_items_indices_original:
        if item_original not in unique_items_indices_training:
            training_indices.append(df_new[df_new["item_index"]==item_original].index.tolist()[0]) # For each item not inclcuded in the training set, select its most early interaction to the training set.
    df_train=df_new.loc[training_indices]
    df_complement=df_new[~df_new.index.isin(training_indices)]
    # Note: The training set should include interactions that involve all the users and items by now.
    for user in unique_users_indices_original:
        num_interactions_total=df_new[df_new["user_index"]==user].shape[0]
        num_interactions_in_df_train=df_train[df_train["user_index"]==user].shape[0]
        interactions_in_df_complement=df_complement[df_complement["user_index"]==user]
        num_interactions_in_df_complement=interactions_in_df_complement.shape[0]
        interaction_indices_in_df_complement=interactions_in_df_complement.index.tolist()
        if num_interactions_in_df_train<num_interactions_total-m:
            index=0
            while num_interactions_in_df_complement>m:
                training_indices.append(interaction_indices_in_df_complement[index])
                index+=1
                num_interactions_in_df_complement-=1
    df_train=df_new.loc[training_indices]
    return df_train
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

def user_item_pair_check(users_indices,item_indices,df_subject):
    """
    Parameters
    ----------
    user_indices: the list of user indices.
    item_indices: the list of item indices.
    df: the subject dataset from which the largest subset that only contains user indices found in user_indices and item indices found in item_indices.
    
    Returns
    --------
    df_subset
    """
    for interaction in df_subject.itertuples():
        user_index=getattr(interaction,"user_index")
        item_index=getattr(interaction,"item_index")
        row_index=getattr(interaction,"Index")
        if (user_index not in users_indices) or (item_index not in item_indices):
            df_subject.drop(axis=0,index=[row_index],inplace=True)
    df_subset=df_subject
    return df_subset
            
        






