#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:07:20 2020

@author: nuoyuan
"""
from Relevant_Set_Construction import relevant_set_construction
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance

def get_unbiasedDCG_per_user(df_test,user_index,average_rating,users_biases,items_biases,users_features,items_features,users_preferences,items_performances,threshold,c):
     relevant_set_user=relevant_set_construction(df_test,user_index,threshold)
     num_relevant_items_user=relevant_set_user.shape[0]
     item_indices_user=df_test[df_test["user_index"]==user_index]["item_index"].values.tolist()
     print(item_indices_user)
     relevance_levels_user=list()
     for item_index in item_indices_user:
         relevance_level_predicted=average_rating+users_biases[user_index]+items_biases[item_index]+np.dot(c*users_features[user_index]+(1-c)*users_preferences[user_index],c*items_features[item_index]+(1-c)*items_performances[item_index])
         relevance_levels_user.append(relevance_level_predicted)
     item_ranking_user=Ranking_Based_on_Relevance.ranking_based_on_relevance(relevance_levels_user)
     df=pd.DataFrame({"item_ranking_user":item_ranking_user},index=item_indices_user)
     print(df[df.index.duplicated()])
     unbiased_DCG_user=0
     if num_relevant_items_user!=0:
         for relevant_item_index in relevant_set_user["item_index"].values.tolist():
             relevant_item_ranking=df.loc[relevant_item_index,"item_ranking_user"]
             expo_prob_item=relevant_set_user[relevant_set_user["item_index"]==relevant_item_index]["exposure_probability"].values[0]
             unbiased_DCG_item=1/(np.log2(relevant_item_ranking+1)*expo_prob_item)
             unbiased_DCG_user+=unbiased_DCG_item
         unbiased_DCG_user=unbiased_DCG_user/num_relevant_items_user
     return unbiased_DCG_user
     