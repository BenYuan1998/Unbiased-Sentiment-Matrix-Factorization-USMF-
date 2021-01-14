# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:00:37 2020

@author: Administrator
"""

import sys
path_general_functions=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_functions)

from Renew_Set_MF_Opin import renew_set_MF_Opin
from TrainingAlgorithm_MF_Opin import trainingalgorithm_MF_Opin
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Get_UnbiasedDCG
import Get_UnbiasedRecall_MF_Opin


class MF_Opin(object):
    def __init__(self,df_train,df_test,expo_prob_train,expo_prob_test,opin_scores_train,opin_scores_test):
        """
        Parameters
        ----------
        df_train : the training set stored as a Pandas dataframe
        df_test : the test set stored as a Pandas dataframe
        expo_prob_train: the propensity score estimates for all the user-item interactions in the training set stored as a list
        expo_prob_test: the propensity score estimates for all the user-item interactions in the test set stored as a list
        opin_scores_train: the opinion scores for all the user-item interactions in the training set stored as a list
        opin_scores_test: the opinion scores for all the user-item interactions in the test set stored as a list
        Returns
        -------
        None.
        """
        self.df_train=df_train
        self.df_test=df_test
        self.expo_prob_train=expo_prob_train
        self.expo_prob_test=expo_prob_test
        self.opin_scores_train=opin_scores_train
        self.opin_scores_test=opin_scores_test
    def performance_unbiasedNDCG(self,optimal_hp_combination,step=100):
        c=optimal_hp_combination[2]
        df_train=renew_set_MF_Opin(self.df_train,self.expo_prob_train,self.opin_scores_train,c)
        df_test=renew_set_MF_Opin(self.df_test,self.expo_prob_test,self.opin_scores_test,c)              
        # Train the USMF model with the optimal hyperparameter combination on the training set
        trainingalgorithm=trainingalgorithm_MF_Opin(optimal_hp_combination)
        trainingalgorithm.sgd(df_train,step)
        # Assess the model's performance on unbiased DCG
        unbiased_DCG_sum_over_users=0
        unbiased_DCG_ideal_sum_over_users=0
        unique_user_indices_test=np.unique(df_test["user_index"].values).tolist()
        num_users=len(unique_user_indices_test)
        for useri in unique_user_indices_test:
            df_useri_interactions_test=df_test[df_test["user_index"]==useri]
            num_items=df_useri_interactions_test.shape[0]
            useri_list_relevance_predicted=list()
            useri_list_relevance_actual=list()
            useri_list_expo_prob=list()
            for useri_item_interaction in df_useri_interactions_test.itertuples():
                user_index=useri
                item_index=getattr(useri_item_interaction,"item_index")
                expo_prob=getattr(useri_item_interaction,"exposure_probability")
                relevance_actual=getattr(useri_item_interaction,"relevance")
                relevance_predicted=np.dot(trainingalgorithm.user_feature[user_index],trainingalgorithm.item_feature[item_index])
                useri_list_relevance_actual.append(relevance_actual)
                useri_list_relevance_predicted.append(relevance_predicted)
                useri_list_expo_prob.append(expo_prob)
            useri_list_ranking_predicted=Ranking_Based_on_Relevance.ranking_based_on_relevance(useri_list_relevance_predicted)
            useri_list_ranking_actual=Ranking_Based_on_Relevance.ranking_based_on_relevance(useri_list_relevance_actual)
            unbiased_DCG=Get_UnbiasedDCG.get_unbiasedDCG(useri_list_ranking_predicted,useri_list_relevance_actual,useri_list_expo_prob)
            unbiased_DCG_ideal=Get_UnbiasedDCG.get_unbiasedDCG(useri_list_ranking_actual,useri_list_relevance_actual,useri_list_expo_prob)
            unbiased_DCG_average_over_items=unbiased_DCG/num_items
            unbiased_DCG_ideal_average_over_items=unbiased_DCG_ideal/num_items
            unbiased_DCG_sum_over_users+=unbiased_DCG_average_over_items
            unbiased_DCG_ideal_sum_over_users+=unbiased_DCG_ideal_average_over_items
        unbiased_DCG_average_over_users=unbiased_DCG_sum_over_users/num_users
        unbiased_DCG_ideal_average_over_users=unbiased_DCG_ideal_sum_over_users/num_users
        unbiased_NDCG=unbiased_DCG_average_over_users/unbiased_DCG_ideal_average_over_users
        return unbiased_NDCG
    def performance_unbiasedrecall(self,optimal_hp_combination,k,step=100):
        c=optimal_hp_combination[2]
        df_train=renew_set_MF_Opin(self.df_train,self.expo_prob_train,self.opin_scores_train,c)
        df_test=renew_set_MF_Opin(self.df_test,self.expo_prob_test,self.opin_scores_test,c) 
        # Train the USMF model with the optimal hyperparameter combination on the training set
        trainingalgorithm=trainingalgorithm_MF_Opin(optimal_hp_combination)
        trainingalgorithm.sgd(df_train,step)
        user_feature=trainingalgorithm.user_feature
        item_feature=trainingalgorithm.item_feature
        # Assess the model's performance on unbiased Recall
        unbiased_recall_sum_over_users=0
        unique_user_indices_test=np.unique(df_test["user_index"].values).tolist()
        num_users=len(unique_user_indices_test)
        for useri in unique_user_indices_test:
            threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
            unbiased_recall_useri=Get_UnbiasedRecall_MF_Opin.get_unbiasedrecall_MF_Opin(df_test,useri,user_feature,item_feature,threshold,k)
            unbiased_recall_sum_over_users+=unbiased_recall_useri
        unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
        return unbiased_recall_average_over_users