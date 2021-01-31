#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:57:40 2020

@author: nuoyuan
"""

def renew_set(df_train_old,expo_prob):
    """
    Parameters
    ----------
    df_train_old : The dataframe with userindex, itemindex, initial relevance prediction based on numerical rating as the entries for each row
    expo_prob : The estimated exposure probabilities for all the training samples stored as a list
    Returns
    -------
    df_train_new: The modified traing set stored as a dataframe with userindex, itemindex, modified rating, and exposure probability as the entries for each row

    """
    import numpy as np
    columns=["user_index","item_index","rating"]
    df_train_new=df_train_old[columns]
    df_train_new["exposure_probability"]=expo_prob
    return df_train_new











                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
