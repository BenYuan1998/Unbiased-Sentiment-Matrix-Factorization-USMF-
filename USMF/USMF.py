# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:32:58 2021

@author: Administrator
"""

import sys
path_general_functions="../General functions"
sys.path.append(path_general_functions)

from Renew_Set import renew_set
from Training_Algorithm import trainingalgorithm_USMF
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Get_UnbiasedDCG
import Get_UnbiasedRecall


class USMF(object):
    def __init__(self,df_train,df_test,expo_prob_train,expo_prob_test,X_train_validation,Y_train_validation):
        """
        Parameters
        ----------
        df_train : the training set stored as a Pandas dataframe
        df_test : the test set stored as a Pandas dataframe
        expo_prob_train: the propensity score estimates for all the user-item interactions in the training set stored as a list
        expo_prob_test: the propensity score estimates for all the user-item interactions in the test set stored as a list
        X: the user-feature attention matrix
        Y: the item-feature quality matrix
        Returns
        -------
        None.
        """
        self.df_train=df_train
        self.df_test=df_test
        self.expo_prob_train=expo_prob_train
        self.expo_prob_test=expo_prob_test
        self.X_train_validation=X_train_validation
        self.Y_train_validation=Y_train_validation
    def performance_unbiasedRMSE(self,optimal_hp_combination,step=100):
        c=optimal_hp_combination[2]
        df_train=renew_set(df_train_old=self.df_train, expo_prob=self.expo_prob_train)
        df_test=renew_set(df_train_old=self.df_test, expo_prob=self.expo_prob_test)
        # Train the USMF model with the given hyperparameter combination on the training set.
        average_rating=np.mean(self.df_train["rating"].values)
        trainingalgorithm=trainingalgorithm_USMF(optimal_hp_combination,average_rating)
        [iterations,costs]=trainingalgorithm.sgd(df_train,self.X_train_validation,self.Y_train_validation,step=100)
        # Assess the model's performance on unbiased RMSE
        MSE_sum_over_items=0
        unique_item_indices_test=np.unique(df_test["item_index"].values).tolist()
        num_items=len(unique_item_indices_test)
        for itemi in unique_item_indices_test:
            SE_itemi=0
            df_test_itemi=df_test[df_test["item_index"]==itemi]
            num_interactions=df_test_itemi.shape[0]
            for itemi_interaction in df_test_itemi.itertuples():
                user_index=getattr(itemi_interaction,"user_index")
                item_index=itemi
                user_preference=self.X_train_validation[user_index]
                item_performance=self.Y_train_validation[item_index]
                relevance_actual=getattr(itemi_interaction,"rating")
                relevance_predicted=average_rating+trainingalgorithm.user_bias[user_index]+trainingalgorithm.item_bias[item_index]+np.dot(c*trainingalgorithm.user_feature[user_index]+(1-c)*user_preference,c*trainingalgorithm.item_feature[item_index]+(1-c)*item_performance)
                square_error=(relevance_actual-relevance_predicted)**2
                SE_itemi+=square_error
            MSE_itemi=SE_itemi/num_interactions
            MSE_sum_over_items+=MSE_itemi
        unbiased_MSE=MSE_sum_over_items/num_items
        return unbiased_MSE
    def performance_unbiasedDCG(self,optimal_hp_combination,step=100):
        c=optimal_hp_combination[2]
        df_train=renew_set(df_train_old=self.df_train, expo_prob=self.expo_prob_train)
        df_test=renew_set(df_train_old=self.df_test,expo_prob=self.expo_prob_test)
        average_rating=np.mean(self.df_train["rating"].values)
        # Train the USMF model with the given hyperparameter combination on the training set.
        trainingalgorithm=trainingalgorithm_USMF(optimal_hp_combination,average_rating)
        [iterations,costs]=trainingalgorithm.sgd(df_train,self.X_train_validation,self.Y_train_validation,step=100)
        # Assess the model's performance on unbiased NDCG
        users_biases=trainingalgorithm.user_bias
        items_biases=trainingalgorithm.item_bias
        users_features=trainingalgorithm.user_feature
        items_features=trainingalgorithm.item_feature
        users_preferences=self.X_train_validation
        items_performances=self.Y_train_validation
        unbiased_DCG_sum_over_users=0
        unique_user_indices_test=np.unique(df_test["user_index"].values).tolist()
        num_users=len(unique_user_indices_test)
        for useri in unique_user_indices_test:
            threshold=3
            unbiased_DCG_useri=Get_UnbiasedDCG.get_unbiasedDCG_per_user(df_test,useri,average_rating,users_biases,items_biases,users_features,items_features,users_preferences,items_performances,threshold,c)
            unbiased_DCG_sum_over_users+=unbiased_DCG_useri
        unbiased_DCG=unbiased_DCG_sum_over_users/num_users
        return unbiased_DCG
    def performance_unbiasedrecall(self,optimal_hp_combination,k,step=100):
        c=optimal_hp_combination[2]
        average_rating=np.mean(self.df_train["rating"].values)
        df_train=renew_set(df_train_old=self.df_train, expo_prob=self.expo_prob_train)
        df_test=renew_set(df_train_old=self.df_test, expo_prob=self.expo_prob_test)
        # Train the USMF model with the given hyperparameter combination on the training set.
        trainingalgorithm=trainingalgorithm_USMF(optimal_hp_combination,average_rating)
        [iterations,costs]=trainingalgorithm.sgd(df_train,self.X_train_validation,self.Y_train_validation,step=100)
        users_biases=trainingalgorithm.user_bias
        items_biases=trainingalgorithm.item_bias
        users_features=trainingalgorithm.user_feature
        items_features=trainingalgorithm.item_feature
        users_preferences=self.X_train_validation
        items_performances=self.Y_train_validation
        # Assess the model's performance on unbiased Recall
        unbiased_recall_sum_over_users=0
        unique_user_indices_test=np.unique(df_test["user_index"].values).tolist()
        num_users=len(unique_user_indices_test)
        for useri in unique_user_indices_test:
            threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
            # Test the model's performance on the top-5 recommendation task via unbiased recall.
            unbiased_recall_useri=Get_UnbiasedRecall.get_unbiasedrecall(df_test,useri,average_rating,users_biases,items_biases,users_features,items_features,users_preferences,items_performances,threshold,c,k)
            unbiased_recall_sum_over_users+=unbiased_recall_useri
        unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
        return unbiased_recall_average_over_users