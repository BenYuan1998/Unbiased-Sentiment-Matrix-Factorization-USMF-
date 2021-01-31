# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:41:15 2021

@author: Ben1998
"""
def txt_to_dataframe(dataset_path):
    import pandas as pd
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
    rating_list=[]
    with open(dataset_path,"r") as f:
        for interaction in f.readlines():
            interaction=eval(interaction.strip("\n"))
            if len(interaction)<5:continue
            user=interaction[0]
            item=interaction[1]
            rating=interaction[3]
            user_list.append(user)
            item_list.append(item)
            rating_list.append(rating)
    data["userId"]=user_list
    data["songId"]=item_list
    data["rating"]=rating_list
    df=pd.DataFrame.from_dict(data,orient="columns")
    df.dropna(axis=0,how="any",inplace=True)
    return df

def training_set(df_new,m):
    import numpy as np
    """
    Parameters
    ----------
    df_new: the subject dataset stored as a dataframe
    m: the number of most recent interactions for each user to be included in the test/validation set.
    Returns
    -------
    df_train: the training set stored as a dataframe
   """
    unique_users_indices_original=np.unique(df_new["userId"].values).tolist()
    unique_items_indices_original=np.unique(df_new["songId"].values).tolist()
    training_indices=list()
    for user in unique_users_indices_original:
        interacted_items_indices=df_new[df_new["userId"]==user].index.tolist()
        training_indices.append(interacted_items_indices[0]) # For user in the original dataset, select his/her most early interaction to the training set.
    unique_items_indices_training=np.unique(df_new.loc[training_indices]["songId"].values).tolist()
    for item_original in unique_items_indices_original:
        if item_original not in unique_items_indices_training:
            training_indices.append(df_new[df_new["songId"]==item_original].index.tolist()[0]) # For each item not inclcuded in the training set, select its most early interaction to the training set.
    df_train=df_new.loc[training_indices]
    df_complement=df_new[~df_new.index.isin(training_indices)]
    # Note: The training set should include interactions that involve all the users and items by now.
    for user in unique_users_indices_original:
        num_interactions_total=df_new[df_new["userId"]==user].shape[0]
        num_interactions_in_df_train=df_train[df_train["userId"]==user].shape[0]
        interactions_in_df_complement=df_complement[df_complement["userId"]==user]
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