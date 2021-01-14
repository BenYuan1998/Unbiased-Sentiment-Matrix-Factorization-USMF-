# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:29:51 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Relevant_Set_Construction_EFM
import Ranking_Score

def get_unbiasedrecall_EFM(df_test_actual,df_test_predicted,user_index,threshold,k):
    """
    Parameters
    ----------
    df_test_actual: the test set stored as a Pandas dataframe with the ground-truth relevance levels for the column "relevance".
    df_test_predicted: the test set stored as a Pandas dataframe with the predicted relevance levels for the column "relevance".
    user_index: The index of the user of interest.
    threshold:  The cut-off point for determining relevant and irrelevant items. Note: The conventional threshold is 3/4 on a five-point scale.
    k: the number of most highly ranked items for each user considered for unbiased recall. 
    Returns
    -------
    recall: The recall for the user's top-k recommendation list.
    """
    user_items_indices_test=df_test_actual[df_test_actual["user_index"]==user_index]["item_index"].values.tolist()
    user_item_interactions=df_test_predicted[df_test_predicted["user_index"]==user_index]
    user_relevance_level_predicted_test=list()
    for item_index in user_items_indices_test:
        relevance_level_predicted=user_item_interactions[user_item_interactions["item_index"]==item_index]["relevance"].values[0]
        user_relevance_level_predicted_test.append(relevance_level_predicted)
    user_ranking_predicted_test=Ranking_Based_on_Relevance.ranking_based_on_relevance(user_relevance_level_predicted_test)
    df=pd.DataFrame({"item_ranking_predicted":user_ranking_predicted_test},index=user_items_indices_test)
    relevant_set=Relevant_Set_Construction_EFM.relevant_set_construction_EFM(df_test_actual,user_index,threshold)
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
