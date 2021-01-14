#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:49:20 2020

@author: nuoyuan
"""
import sys
path_general_functions=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_functions)

from Renew_Set_USMF import renew_set_USMF
from TrainingAlgorithm_USMF import trainingalgorithm_USMF
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Get_UnbiasedDCG
import Get_UnbiasedRecall_USMF


class USMF(object):
    def __init__(self,df_train,df_test,expo_prob_train,expo_prob_test,opin_scores_train):
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
    def performance_unbiasedRMSE(self,optimal_hp_combination,step=100):
         for index,hp_combination in enumerate(self.hp_combinations):
            c=optimal_hp_combination[2]
            df_train=renew_set_USMF(df_train_old=self.df_train,expo_prob=self.expo_prob_train,opin_scores=self.opin_scores_train,c=c)
            df_test=self.df_test
            df_test.rename(columns={"rating":"relevance"},inplace=True)
            df_test["exposure_probability"]=self.expo_prob_test
            # Train the USMF model with the given hyperparameter combination on the training set.
            trainingalgorithm=trainingalgorithm_USMF(optimal_hp_combination)
            trainingalgorithm.sgd(df_train)
            # Assess the model's performance on unbiased RMSE
            SE_sum_over_items=0
            unique_item_indices_test=np.unique(df_test["user_index"].values).tolist()
            num_items=len(unique_item_indices_test)
            for itemi in unique_item_indices_test:
                SE_itemi=0
                df_test_itemi=df_test[df_test["item_index"]==itemi]
                for itemi_interaction in df_test_itemi.itertuples():
                    user_index=getattr(itemi_interaction,"user_index")
                    item_index=itemi
                    relevance_actual=getattr(itemi_interaction,"relevance")
                    relevance_predicted=np.dot(trainingalgorithm.user_feature[user_index],trainingalgorithm.item_feature[item_index])
                    square_error=(relevance_actual-relevance_predicted)**2
                    SE_itemi+=square_error
                SE_sum_over_items+=SE_itemi
            unbiased_RMSE=np.sqrt(SE_sum_over_items/num_items)
            return unbiased_RMSE
    def performance_unbiasedNDCG(self,optimal_hp_combination,step=100):
        c=optimal_hp_combination[2]
        df_train=renew_set_USMF(df_train_old=self.df_train,expo_prob=self.expo_prob_train,opin_scores=self.opin_scores_train,c=c)
        df_test=self.df_test
        df_test.rename(columns={"rating":"relevance"},inplace=True)
        df_test["exposure_probability"]=self.expo_prob_test             
        # Train the USMF model with the optimal hyperparameter combination on the training set
        trainingalgorithm=trainingalgorithm_USMF(optimal_hp_combination)
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
        df_train=renew_set_USMF(df_train_old=self.df_train,expo_prob=self.expo_prob_train,opin_scores=self.opin_scores_train,c=c)
        df_test=self.df_test
        df_test.rename(columns={"rating":"relevance"},inplace=True)
        df_test["exposure_probability"]=self.expo_prob_test
        # Train the USMF model with the optimal hyperparameter combination on the training set
        trainingalgorithm=trainingalgorithm_USMF(optimal_hp_combination)
        trainingalgorithm.sgd(df_train,step)
        user_feature=trainingalgorithm.user_feature
        item_feature=trainingalgorithm.item_feature
        # Assess the model's performance on unbiased Recall
        unbiased_recall_sum_over_users=0
        unique_user_indices_test=np.unique(df_test["user_index"].values).tolist()
        num_users=len(unique_user_indices_test)
        for useri in unique_user_indices_test:
            threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
            k=5 # Test the model's performance on the top-5 recommendation task via unbiased recall.
            unbiased_recall_useri=Get_UnbiasedRecall_USMF.get_unbiasedrecall_USMF(df_test,useri,user_feature,item_feature,threshold,k)
            unbiased_recall_sum_over_users+=unbiased_recall_useri
        unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
        return unbiased_recall_average_over_users
        
        