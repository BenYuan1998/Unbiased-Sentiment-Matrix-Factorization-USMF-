# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:42:03 2020

@author: Administrator
"""
import numpy as np
def ranking_score(user_index,item_index,X,Y,A,N,k,c):
    """
    Parameters:
    ----------
    user_index: the index of some user.
    item_index: the index of some item.
    X: the user-feature attention matrix.
    Y: the item-feature quality matrix. 
    A：the user-item rating matrix.
    N: the largest possible value on the numerical rating scale.
    k: the number of most cared item features.
    c: the weighing scalar. 
    Note: For generating predicted ranking scores, it should be noted that: 
        
    Returns:
    -------
    R: the ranking score of the subject item for the subject user.
    """
    idx=np.argpartition(Y[item_index],-k)
    idx=idx[-k:]
    R=c*(np.dot(X[user_index,idx],Y[item_index,idx])/(k*N))+(1-c)*A[user_index,item_index]
    return R
    