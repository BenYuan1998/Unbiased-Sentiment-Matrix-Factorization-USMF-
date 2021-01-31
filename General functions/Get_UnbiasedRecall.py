# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:21:02 2020

@author: Administrator
"""
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Relevant_Set_Construction

def get_unbiasedrecall(df_test,user_index,average_rating,users_biases,items_biases,users_features,items_features,users_preferences,items_performances,threshold,c,k):
    user_items_indices_test=df_test[df_test["user_index"]==user_index]["item_index"].values.tolist()
    user_relevance_level_predicted_test=list()
    for item_index in user_items_indices_test:
        relevance_level_predicted=average_rating+users_biases[user_index]+items_biases[item_index]+np.dot(c*users_features[user_index]+(1-c)*users_preferences[user_index],c*items_features[item_index]+(1-c)*items_performances[item_index])
        user_relevance_level_predicted_test.append(relevance_level_predicted)
    user_ranking_predicted_test=Ranking_Based_on_Relevance.ranking_based_on_relevance(user_relevance_level_predicted_test)
    df=pd.DataFrame({"item_ranking_predicted":user_ranking_predicted_test},index=user_items_indices_test)
    relevant_set=Relevant_Set_Construction.relevant_set_construction(df_test,user_index,threshold)
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
                
                
    
    