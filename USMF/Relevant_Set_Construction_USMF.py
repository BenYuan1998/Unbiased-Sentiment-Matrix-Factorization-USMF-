# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:57:50 2020

@author: Administrator
"""

import pandas as pd

def relevant_set_construction_USMF(df_test,user_index,threshold):
    """
    Parameters
    ----------
    df_test : The test set stored as a Pandas dataframe.
    user_index : The index of the user of interest.
    threshold : The cut-off point for determining relevant and irrelevant items. Note: The conventional threshold is 3/4 on a five-point scale. 

    Returns
    -------
    df_relevant：The set of relevant items for the user of interest. If all the items in the test set for the user are deemed irrelvant by the threshold, 
    df_relevant=None
    """
    user_items_interactions=df_test[df_test["user_index"]==user_index]
    relevant_indices=list()
    for index, interaction in user_items_interactions.iterrows():
        relevance_level=interaction["relevance"]
        if relevance_level>=threshold:
            relevant_indices.append(index)
    if len(relevant_indices)==0:
        df_relevant=pd.DataFrame()
    else:
        df_relevant=df_test.loc[relevant_indices]
    return df_relevant




