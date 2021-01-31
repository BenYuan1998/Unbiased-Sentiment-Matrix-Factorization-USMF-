# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:56:55 2020

@author: Administrator
"""
import sys
path_USMF="../USMF"
sys.path.append(path_USMF)
path_train_test_split="../Data pre-processing"
sys.path.append(path_train_test_split)
path_general_functions="../General functions"
sys.path.append(path_general_functions)



from Renew_Set import renew_set
from Training_Algorithm import trainingalgorithm_USMF
import numpy as np
import pandas as pd
import Get_UnbiasedDCG
import Get_UnbiasedRecall
from Checking_Convergence import checking_convergence

class hyperparametertuning_train_validation_split_USMF(object):
    def __init__(self,df_train,df_validation,expo_prob_train,expo_prob_validation,X_train,Y_train,hp_combinations):
        """
        Parameters
        ----------
        df_train : the training set stored as a Pandas dataframe.
        df_validation: the validation set stored as a Pandas dataframe.
        expo_prob : the propensity score estimations for all the user-item samples in the training set stored as a list.
        X: the user-feature attention matrix.
        Y: the item-feature quality matrix.
        hp_combinations: all the hyperparameter combinations stored  as a numpy array
        Returns
        -------
        None.
        """
        self.df_train=df_train=df_train
        self.df_validation=df_validation
        self.expo_prob_train=expo_prob_train
        self.expo_prob_validation=expo_prob_validation
        self.X_train=X_train
        self.Y_train=Y_train
        self.hp_combinations=hp_combinations
        self.hp_combination_performance_unbiasedMSE=list()
        self.hp_combination_performance_unbiasedDCG=list()
        self.hp_combination_performance_unbiasedrecall=list()
    def train_validation_split_unbiasedMSE(self):
        for index,hp_combination in enumerate(self.hp_combinations):
            c=hp_combination[2]
            df_train_hptuning=renew_set(df_train_old=self.df_train, expo_prob=self.expo_prob_train)
            df_validation_hptuning=renew_set(df_train_old=self.df_validation, expo_prob=self.expo_prob_validation)
            # Train the USMF model with the given hyperparameter combination on the training set.
            average_rating=np.mean(self.df_train["rating"].values)
            trainingalgorithm=trainingalgorithm_USMF(hp_combination,average_rating)
            [iterations,costs]=trainingalgorithm.sgd(df_train_hptuning,self.X_train,self.Y_train,step=100)
            if index==0:
                print(costs)
                checking_convergence(iterations,costs)
            # Assess the model's performance on unbiased RMSE
            MSE_sum_over_items=0
            unique_item_indices_validation=np.unique(df_validation_hptuning["item_index"].values).tolist()
            num_items=len(unique_item_indices_validation)
            for itemi in unique_item_indices_validation:
                SE_itemi=0
                df_validation_hptuning_itemi=df_validation_hptuning[df_validation_hptuning["item_index"]==itemi]
                num_interactions=df_validation_hptuning_itemi.shape[0]
                for itemi_interaction in df_validation_hptuning_itemi.itertuples():
                    user_index=getattr(itemi_interaction,"user_index")
                    item_index=itemi
                    user_preference=self.X_train[user_index]
                    item_performance=self.Y_train[item_index]
                    relevance_actual=getattr(itemi_interaction,"rating")
                    relevance_predicted=average_rating+trainingalgorithm.user_bias[user_index]+trainingalgorithm.item_bias[item_index]+np.dot(c*trainingalgorithm.user_feature[user_index]+(1-c)*user_preference,c*trainingalgorithm.item_feature[item_index]+(1-c)*item_performance)
                    square_error=(relevance_actual-relevance_predicted)**2
                    SE_itemi+=square_error
                MSE_itemi=SE_itemi/num_interactions
                MSE_sum_over_items+=MSE_itemi
            unbiased_MSE=MSE_sum_over_items/num_items
            self.hp_combination_performance_unbiasedMSE.append(unbiased_MSE)
    def train_validation_split_unbiasedDCG(self):
        for index,hp_combination in enumerate(self.hp_combinations):
            c=hp_combination[2]
            df_train_hptuning=renew_set(df_train_old=self.df_train, expo_prob=self.expo_prob_train)
            df_validation_hptuning=renew_set(df_train_old=self.df_validation,expo_prob=self.expo_prob_validation)
            average_rating=np.mean(self.df_train["rating"].values)
            # Train the USMF model with the given hyperparameter combination on the training set.
            trainingalgorithm=trainingalgorithm_USMF(hp_combination,average_rating)
            [iterations,costs]=trainingalgorithm.sgd(df_train_hptuning,self.X_train,self.Y_train,step=100)
            if index==0:
                checking_convergence(iterations,costs)
            # Assess the model's performance on unbiased DCG
            users_biases=trainingalgorithm.user_bias
            items_biases=trainingalgorithm.item_bias
            users_features=trainingalgorithm.user_feature
            items_features=trainingalgorithm.item_feature
            users_preferences=self.X_train
            items_performances=self.Y_train
            unbiased_DCG_sum_over_users=0
            unique_user_indices_validation=np.unique(df_validation_hptuning["user_index"].values).tolist()
            num_users=len(unique_user_indices_validation)
            for useri in unique_user_indices_validation:
                threshold=3
                unbiased_DCG_useri=Get_UnbiasedDCG.get_unbiasedDCG_per_user(df_validation_hptuning,useri,average_rating,users_biases,items_biases,users_features,items_features,users_preferences,items_performances,threshold,c)
                unbiased_DCG_sum_over_users+=unbiased_DCG_useri
            unbiased_DCG=unbiased_DCG_sum_over_users/num_users
            self.hp_combination_performance_unbiasedDCG.append(unbiased_DCG)         
    def train_validation_split_unbiasedrecall(self,k):
        for index,hp_combination in enumerate(self.hp_combinations):
            c=hp_combination[2]
            average_rating=np.mean(self.df_train["rating"].values)
            df_train_hptuning=renew_set(df_train_old=self.df_train, expo_prob=self.expo_prob_train)
            df_validation_hptuning=renew_set(df_train_old=self.df_validation, expo_prob=self.expo_prob_validation)
            # Train the USMF model with the given hyperparameter combination on the training set.
            trainingalgorithm=trainingalgorithm_USMF(hp_combination,average_rating)
            [iterations,costs]=trainingalgorithm.sgd(df_train_hptuning,self.X_train,self.Y_train,step=100)
            if index==0:
                checking_convergence(iterations,costs)
            users_biases=trainingalgorithm.user_bias
            items_biases=trainingalgorithm.item_bias
            users_features=trainingalgorithm.user_feature
            items_features=trainingalgorithm.item_feature
            users_preferences=self.X_train
            items_performances=self.Y_train
            # Assess the model's performance on unbiased Recall
            unbiased_recall_sum_over_users=0
            unique_user_indices_validation=np.unique(df_validation_hptuning["user_index"].values).tolist()
            num_users=len(unique_user_indices_validation)
            for useri in unique_user_indices_validation:
                threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
                # Test the model's performance on the top-5 recommendation task via unbiased recall.
                unbiased_recall_useri=Get_UnbiasedRecall.get_unbiasedrecall(df_validation_hptuning,useri,average_rating,users_biases,items_biases,users_features,items_features,users_preferences,items_performances,threshold,c,k)
                unbiased_recall_sum_over_users+=unbiased_recall_useri
            unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
            self.hp_combination_performance_unbiasedrecall.append(unbiased_recall_average_over_users)