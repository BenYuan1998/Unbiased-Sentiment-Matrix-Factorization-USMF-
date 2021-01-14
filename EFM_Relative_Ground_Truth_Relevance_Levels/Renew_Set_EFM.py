# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:22:46 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import Ranking_Score

def renew_set_EFM(df_old,expo_prob,X,Y,A,N,k,c):
    """
    parameters:
    -----------
    df_old: The dataframe with user_index, item_index, review, rating, and date as the dataframe' column indexers.
    expo_prob:The estimated exposure probabilities for all the items contained in df_train_old.
    path_ranking_score: The absolute path of the py.file that stores the function for ranking score generation.
    X: the user-feature attention matrix.
    Y: the item-feature quality matrix. 
    A：the user-item rating matrix.
    N: the largest possible value on the numerical rating scale.
    k: the number of most cared item features.
    c: the weighing scalar. 
    returns: 
    --------
    df_new: The dataframe with user_index, item_index, ranking_score, and exposure_probability as the dataframe's column indexers. 
    """
    columns=["user_index","item_index"]
    df_new=df_old[columns]
    # ranking score generation
    R=list()
    for interaction in df_new.itertuples():
        user=getattr(interaction,"user_index")
        item=getattr(interaction,"item_index")
        rij=Ranking_Score.ranking_score(user_index=user,item_index=item,X=X,Y=Y,A=A,N=N,k=k,c=c)
        R.append(rij)
    df_new["relevance"]=R
    df_new["exposure_probability"]=expo_prob
    return df_new
        
        
    
    
