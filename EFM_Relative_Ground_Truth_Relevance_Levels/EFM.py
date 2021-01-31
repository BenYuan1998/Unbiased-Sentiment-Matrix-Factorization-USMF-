# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:09:50 2021

@author: Administrator
"""

import sys
path_EFM="./"
sys.path.append(path_EFM)
path_train_test_split="../../Data pre-processing"
sys.path.append(path_train_test_split)
path_general_functions="../../General functions"
sys.path.append(path_general_functions)

from Renew_Set import renew_set
from TrainingAlgorithm_EFM import trainingalgorithm_EFM
import numpy as np
import pandas as pd
import Get_UnbiasedDCG_EFM
import Get_UnbiasedRecall_EFM

class EFM(object):
    def __init__(self,df_train,df_test,A_train,A_test,X_train,Y_train,expo_prob_test):
        self.df_train=df_train
        self.df_test=df_test
        self.A_train=A_train
        self.A_test=A_test
        self.X_train=X_train
        self.Y_train=Y_train
        self.expo_prob_test=expo_prob_test
    def performance_unbiasedMSE(self,optimal_hp_combination):
        lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(optimal_hp_combination)
        r=int(r)
        r_=int(r_)
        k=int(k)
        # Variable "matrices" has the following form:[U1,U2,V,H1,H2]
        matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
        U1=matrices[0]
        U2=matrices[1]
        H1=matrices[3]
        H2=matrices[4]
        # compute X_,Y_,A_
        A_=U1.dot(U2.T)+H1.dot(H2.T)
        # Assess the model's performance on unbiased RMSE
        MSE_sum_over_items=0
        unique_item_indices_test=np.unique(self.df_test["item_index"].values).tolist()
        num_items=len(unique_item_indices_test)
        for itemi in unique_item_indices_test:
            SE_itemi=0
            df_test_itemi=self.df_test[self.df_test["item_index"]==itemi]
            num_interactions=df_test_itemi.shape[0]
            for interaction in df_test_itemi.itertuples():
                user_index=getattr(interaction,"user_index")
                item_index=itemi
                relevance_predicted=A_[user_index,item_index]
                relevance_actual=self.A_test[user_index,item_index]
                square_error=(relevance_predicted-relevance_actual)**2
                SE_itemi+=square_error
            MSE_itemi=SE_itemi/num_interactions
            MSE_sum_over_items+=MSE_itemi
        unbiased_MSE=MSE_sum_over_items/num_items
        return unbiased_MSE
    def performance_unbiasedNDCG(self,optimal_hp_combination):
         lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(optimal_hp_combination)
         r=int(r)
         r_=int(r_)
         k=int(k)
         df_test=renew_set(self.df_test,self.expo_prob_test)
         matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
         U1=matrices[0]
         U2=matrices[1]
         H1=matrices[3]
         H2=matrices[4]
         # compute A_
         A_=U1.dot(U2.T)+H1.dot(H2.T)
         # Assess the model's performance on unbiased DCG
         unbiased_DCG_sum_over_users=0
         unique_user_indices_test=np.unique(df_test["user_index"].values).tolist()
         num_users=len(unique_user_indices_test)
         for useri in unique_user_indices_test:
             threshold=3
             unbiased_DCG_useri=Get_UnbiasedDCG_EFM.get_unbiasedDCG_per_user(df_test, useri, A_, threshold)
             unbiased_DCG_sum_over_users+=unbiased_DCG_useri
         unbiased_DCG=unbiased_DCG_sum_over_users/num_users
         return unbiased_DCG
    def performance_unbiasedrecall(self,optimal_hp_combination,k):
        lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(optimal_hp_combination)
        r=int(r)
        r_=int(r_)
        k=int(k)
        df_test=renew_set(self.df_test,self.expo_prob_test)
        matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
        U1=matrices[0]
        U2=matrices[1]
        H1=matrices[3]
        H2=matrices[4]
        # compute A_
        A_=U1.dot(U2.T)+H1.dot(H2.T)
        # Assess the model's performance on unbiased Recall
        unbiased_recall_sum_over_users=0
        unique_user_indices_test=np.unique(df_test["user_index"].values).tolist()
        num_users=len(unique_user_indices_test)
        for useri in unique_user_indices_test:
            threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
            unbiased_recall_useri=Get_UnbiasedRecall_EFM.get_unbiasedrecall_EFM(df_test, useri, A_, threshold,k)
            unbiased_recall_sum_over_users+=unbiased_recall_useri
        unbiased_recall=unbiased_recall_sum_over_users/num_users
        return unbiased_recall