# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:47:59 2020

@author: Administrator
"""

import sys
path_MF_Opin=r"C:\Users\Administrator\Desktop\上科大\代码\Baselines\MF-Opin"
sys.path.append(path_MF_Opin)
path_train_test_split=r"C:\Users\Administrator\Desktop\上科大\代码\Data pre-processing"
sys.path.append(path_train_test_split)
path_general_functions=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_functions)


from Train_Test_Split import training_set
from Train_Test_Split import test_set
from Renew_Set_MF_Opin import renew_set_MF_Opin
from TrainingAlgorithm_MF_Opin import trainingalgorithm_MF_Opin
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Get_UnbiasedDCG
import Get_UnbiasedRecall_MF_Opin


class hyperparametertuning_train_validation_split_MF_Opin(object):
    def __init__(self,df_train,df_validation,expo_prob_train,expo_prob_validation,opin_scores_train,opin_scores_validation,hp_combinations):
        """
        Parameters
        ----------
        df_train : the training set stored as a Pandas dataframe.
        expo_prob : the propensity score estimations for all the user-item samples in the training set stored as a list.
        opin_scores : the opinion scores for all the user-generated reviews contained in the training set stored as a list.
        hp_combinations: all the hyperparameter combinations stored  as a numpy array
        m : the number of most recently occurred user-item interactions for each user to be contained in the validation set.
        Returns
        -------
        None.
        """
        self.df_train=df_train
        self.df_validaiton=df_validation
        self.expo_prob_train=expo_prob_train
        self.expo_prob_validation=expo_prob_validation
        self.opin_scores_train=opin_scores_train
        self.opin_scores_validation=opin_scores_validation
        self.hp_combinations=hp_combinations
        self.hp_combination_performance_unbiasedNDCG=list()
        self.hp_combination_performance_unbiasedrecall=list()
    def train_validation_split_unbiasedNDCG(self,step=100):
        for hp_combination in self.hp_combinations:
            c=hp_combination[2] # The index of the rating-opinion rerlative contribution weight in each hyperparameter combination numpy array is 2.
            df_train_hptuning=renew_set_MF_Opin(self.df_train,self.expo_prob_train,self.opin_scores_train,c)
            df_validation_hptuning=renew_set_MF_Opin(self.df_validaiton,self.expo_prob_validation,self.opin_scores_validation,c)
            trainingalgorithm=trainingalgorithm_MF_Opin(hp_combination)
            # Train the USMF model with the given hyperparameter combination on the training set
            trainingalgorithm.sgd(df_train_hptuning,step)
            # Assess the model's performance on unbiased DCG
            unbiased_DCG_sum_over_users=0
            unbiased_DCG_ideal_sum_over_users=0
            unique_user_indices_validation=np.unique(df_validation_hptuning["user_index"].values).tolist()
            num_users=len(unique_user_indices_validation)
            for useri in unique_user_indices_validation:
                df_validation_hptuning_useri=df_validation_hptuning[df_validation_hptuning["user_index"]==useri]
                num_items=df_validation_hptuning_useri.shape[0]
                useri_list_relevance_predicted=list()
                useri_list_relevance_actual=list()
                useri_list_expo_prob=list()
                for useri_item_interaction in df_validation_hptuning_useri.itertuples():
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
            self.hp_combination_performance_unbiasedNDCG.append(unbiased_NDCG)
    def train_validation_split_unbiasedrecall(self,k,step=100):
        for hp_combination in self.hp_combinations:
            c=hp_combination[2] # The index of the rating-opinion relative contribution weight in each hyperparameter combination numpy array is  2.
            df_train_hptuning=renew_set_MF_Opin(self.df_train,self.expo_prob_train,self.opin_scores_train,c)
            df_validation_hptuning=renew_set_MF_Opin(self.df_validaiton,self.expo_prob_validation,self.opin_scores_validation,c)
            trainingalgorithm=trainingalgorithm_MF_Opin(hp_combination)
            trainingalgorithm.sgd(df_train_hptuning,step)
            user_feature=trainingalgorithm.user_feature
            item_feature=trainingalgorithm.item_feature
            # Assess the model's performance on unbiased Recall
            unbiased_recall_sum_over_users=0
            unique_user_indices_test=np.unique(df_validation_hptuning["user_index"].values).tolist()
            num_users=len(unique_user_indices_test)
            for useri in unique_user_indices_test:
                threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
                unbiased_recall_useri=Get_UnbiasedRecall_MF_Opin.get_unbiasedrecall_MF_Opin(df_validation_hptuning,useri,user_feature,item_feature,threshold,k)
                unbiased_recall_sum_over_users+=unbiased_recall_useri
            unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
            self.hp_combination_performance_unbiasedrecall.append(unbiased_recall_average_over_users)