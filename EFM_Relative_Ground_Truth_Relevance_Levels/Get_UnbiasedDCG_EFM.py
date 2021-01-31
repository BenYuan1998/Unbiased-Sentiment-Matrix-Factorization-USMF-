# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:17:06 2021

@author: Administrator
"""
import sys
path_general_functions="../../General functions"
sys.path.append(path_general_functions)
from Relevant_Set_Construction import relevant_set_construction
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance

def get_unbiasedDCG_per_user(df_test,user_index,A_,threshold):
     relevant_set_user=relevant_set_construction(df_test,user_index,threshold)
     num_relevant_items_user=relevant_set_user.shape[0]
     item_indices_user=df_test[df_test["user_index"]==user_index]["item_index"].values.tolist()
     relevance_levels_user=list()
     for item_index in item_indices_user:
         relevance_level_predicted=A_[user_index,item_index]
         relevance_levels_user.append(relevance_level_predicted)
     item_ranking_user=Ranking_Based_on_Relevance.ranking_based_on_relevance(relevance_levels_user)
     df=pd.DataFrame({"item_ranking_user":item_ranking_user},index=item_indices_user)
     unbiased_DCG_user=0
     if num_relevant_items_user!=0:
         for relevant_item_index in relevant_set_user["item_index"].values.tolist():
             relevant_item_ranking=df.loc[relevant_item_index,"item_ranking_user"]
             expo_prob_item=relevant_set_user[relevant_set_user["item_index"]==relevant_item_index]["exposure_probability"].values[0]
             unbiased_DCG_item=1/(np.log2(relevant_item_ranking+1)*expo_prob_item)
             unbiased_DCG_user+=unbiased_DCG_item
         unbiased_DCG_user=unbiased_DCG_user/num_relevant_items_user
     return unbiased_DCG_user