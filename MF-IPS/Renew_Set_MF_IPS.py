# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:28:03 2020

@author: Administrator
"""

def renew_set_MF_IPS(df_train_old,expo_prob):
    """
    

    Parameters
    ----------
    df_train_old : The dataframe with userindex, itemindex, initial relevance prediction based on numerical rating as the entries for each row
    expo_prob : The estimated exposure probabilities for all the training samples stored as a list
    c : The rating-opinion weight that lies between 0 and 1 and that controls the relative contribution of users' ratings to user-item relevance predictions compared to users' opinions (a hyperparameter)

    Returns
    -------
    df_train_new: The modified traing set stored as a dataframe with userindex, itemindex, modified rating, and exposure probability as the entries for each row

    """
    columns=["user_index","item_index","rating"]
    df_train_new=df_train_old[columns]
    df_train_new["exposure_probability"]=expo_prob
    return df_train_new

