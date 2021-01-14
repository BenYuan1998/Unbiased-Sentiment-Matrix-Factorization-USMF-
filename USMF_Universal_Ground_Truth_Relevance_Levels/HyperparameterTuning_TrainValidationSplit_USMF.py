# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:56:55 2020

@author: Administrator
"""
import sys
path_USMF=r"C:\Users\Administrator\Desktop\上科大\代码\USMF algorithm training"
sys.path.append(path_USMF)
path_train_test_split=r"C:\Users\Administrator\Desktop\上科大\代码\Data pre-processing"
sys.path.append(path_train_test_split)
path_general_functions=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_functions)


from Train_Test_Split import training_set
from Train_Test_Split import test_set
from Renew_Set_USMF import renew_set_USMF
from TrainingAlgorithm_USMF import trainingalgorithm_USMF
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Get_UnbiasedDCG
import Get_UnbiasedRecall_USMF
from Checking_Convergence import checking_convergence

class hyperparametertuning_train_validation_split_USMF(object):
    def __init__(self,df_train,df_validation,expo_prob_train,expo_prob_validation,opin_scores_train,hp_combinations):
        """
        Parameters
        ----------
        df_train : the training set stored as a Pandas dataframe.
        df_validation: the validation set stored as a Pandas dataframe.
        expo_prob : the propensity score estimations for all the user-item samples in the training set stored as a list.
        opin_scores : the opinion scores for all the user-generated reviews contained in the training set stored as a list.
        hp_combinations: all the hyperparameter combinations stored  as a numpy array
        Returns
        -------
        None.
        """
        self.df_train=df_train=df_train
        self.df_validation=df_validation
        self.expo_prob_train=expo_prob_train
        self.expo_prob_validation=expo_prob_validation
        self.opin_scores_train=opin_scores_train
        self.hp_combinations=hp_combinations
        self.hp_combination_peformance_unbiasedRMSE=list()
        self.hp_combination_performance_unbiasedNDCG=list()
        self.hp_combination_performance_unbiasedrecall=list()
    def train_validation_split_unbiasedRMSE(self):
        for index,hp_combination in enumerate(self.hp_combinations):
            c=hp_combination[2]
            df_train_hptuning=renew_set_USMF(df_train_old=self.df_train, expo_prob=self.expo_prob_train, opin_scores=self.opin_scores_train, c=c)
            df_validation_hptuning=self.df_validation
            df_validation_hptuning.rename(columns={"rating":"relevance"},inplace=True)
            df_validation_hptuning["exposure_probability"]=self.expo_prob_validation
            # Train the USMF model with the given hyperparameter combination on the training set.
            trainingalgorithm=trainingalgorithm_USMF(hp_combination)
            [iterations,costs]=trainingalgorithm.sgd(df_train_hptuning,step=500)
            if index==0:
                print(costs)
                checking_convergence(iterations,costs)
            # Assess the model's performance on unbiased RMSE
            SE_sum_over_items=0
            unique_item_indices_validation=np.unique(df_validation_hptuning["user_index"].values).tolist()
            num_items=len(unique_item_indices_validation)
            for itemi in unique_item_indices_validation:
                SE_itemi=0
                df_validation_hptuning_itemi=df_validation_hptuning[df_validation_hptuning["item_index"]==itemi]
                for itemi_interaction in df_validation_hptuning_itemi.itertuples():
                    user_index=getattr(itemi_interaction,"user_index")
                    item_index=itemi
                    relevance_actual=getattr(itemi_interaction,"relevance")
                    relevance_predicted=np.dot(trainingalgorithm.user_feature[user_index],trainingalgorithm.item_feature[item_index])
                    square_error=(relevance_actual-relevance_predicted)**2
                    SE_itemi+=square_error
                SE_sum_over_items+=SE_itemi
            unbiased_RMSE=np.sqrt(SE_sum_over_items/num_items)
            self.hp_combination_performance_unbiasedRMSE.append(unbiased_RMSE)
    def train_validation_split_unbiasedNDCG(self):
        for index,hp_combination in enumerate(self.hp_combinations):
            c=hp_combination[2]
            df_train_hptuning=renew_set_USMF(df_train_old=self.df_train, expo_prob=self.expo_prob_train, opin_scores=self.opin_scores_train,c=c)
            df_validation_hptuning=self.df_validation
            df_validation_hptuning.rename(columns={"rating":"relevance"},inplace=True)
            df_validation_hptuning["exposure_probability"]=self.expo_prob_validation
            # Train the USMF model with the given hyperparameter combination on the training set.
            trainingalgorithm=trainingalgorithm_USMF(hp_combination)
            [iterations,costs]=trainingalgorithm.sgd(df_train_hptuning,step=500)
            if index==0:
                print(costs)
                checking_convergence(iterations,costs)
            # Assess the model's performance on unbiased NDCG
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
    def train_validation_split_unbiasedrecall(self,k):
        for index,hp_combination in enumerate(self.hp_combinations):
            c=hp_combination[2]
            df_train_hptuning=renew_set_USMF(df_train_old=self.df_train, expo_prob=self.expo_prob_train, opin_scores=self.opin_scores_train,c=c)
            df_validation_hptuning=self.df_validation
            df_validation_hptuning.rename(columns={"rating":"relevance"},inplace=True)
            df_validation_hptuning["exposure_probability"]=self.expo_prob_validation
            # Train the USMF model with the given hyperparameter combination on the training set.
            trainingalgorithm=trainingalgorithm_USMF(hp_combination)
            [iterations,costs]=trainingalgorithm.sgd(df_train_hptuning,step=500)
            if index==0:
                checking_convergence(iterations,costs)
            user_feature=trainingalgorithm.user_feature
            item_feature=trainingalgorithm.item_feature
            # Assess the model's performance on unbiased Recall
            unbiased_recall_sum_over_users=0
            unique_user_indices_test=np.unique(df_validation_hptuning["user_index"].values).tolist()
            num_users=len(unique_user_indices_test)
            for useri in unique_user_indices_test:
                threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
                #k=3 # Test the model's performance on the top-5 recommendation task via unbiased recall.
                unbiased_recall_useri=Get_UnbiasedRecall_USMF.get_unbiasedrecall_USMF(df_validation_hptuning,useri,user_feature,item_feature,threshold,k)
                unbiased_recall_sum_over_users+=unbiased_recall_useri
            unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
            self.hp_combination_performance_unbiasedrecall.append(unbiased_recall_average_over_users)