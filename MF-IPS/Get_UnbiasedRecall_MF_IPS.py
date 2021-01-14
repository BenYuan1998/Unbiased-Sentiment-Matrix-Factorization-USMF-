# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:25:32 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Relevant_Set_Construction_MF_IPS

def get_unbiasedrecall_MF_IPS(df_test,user_index,user_feature,item_feature,threshold,k):
    """
    Parameters
    ----------
    df_test: The test set stored as a Pandas dataframe.
    user_index: The index of the user of interest
    user-feature: A Python dictionary for which each user index and his/her corresponding feature constitute a key-value pair.
    item_feature: A Python dictionary for whcih each item index and its corresponding feature constitute a key-value pair.
    threshold:  The cut-off point for determining relevant and irrelevant items. Note: The conventional threshold is 3/4 on a five-point scale.
    k : The size of the recommendation list. 

    Returns
    -------
    recall: The recall for the user's top-k recommendation list.
    """
    user_items_indices_test=df_test[df_test["user_index"]==user_index]["item_index"].values.tolist()
    user_relevance_level_predicted_test=list()
    for item_index in user_items_indices_test:
        relevance_level_predicted=np.dot(user_feature[user_index],item_feature[item_index])
        user_relevance_level_predicted_test.append(relevance_level_predicted)
    user_ranking_predicted_test=Ranking_Based_on_Relevance.ranking_based_on_relevance(user_relevance_level_predicted_test)
    df=pd.DataFrame({"item_ranking_predicted":user_ranking_predicted_test},index=user_items_indices_test)
    relevant_set=Relevant_Set_Construction_MF_IPS.relevant_set_construction_MF_IPS(df_test,user_index,threshold)
    unbiased_recall=0
    if relevant_set.shape[0]!=0:
        relevant_items_indices=relevant_set["item_index"].values
        size_relevant_set=relevant_set.shape[0]
        potential_denominators=np.array([k,size_relevant_set])
        for relevant_item_index in relevant_items_indices:
            relevance=df.loc[relevant_item_index,"item_ranking_predicted"]
            expo_prob=relevant_set[relevant_set["item_index"]==relevant_item_index]["exposure_probability"].values[0]
            if relevance<=k:
                unbiased_recall+=1/(potential_denominators[np.argmin(potential_denominators)]*expo_prob)
    else:
        unbiased_recall=0
    return unbiased_recall
   


