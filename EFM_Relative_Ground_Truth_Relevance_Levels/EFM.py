# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:09:50 2021

@author: Administrator
"""

import sys
path_EFM=r"C:\Users\Administrator\Desktop\上科大\代码\Baselines\EFM"
sys.path.append(path_EFM)
path_train_test_split=r"C:\Users\Administrator\Desktop\上科大\代码\Data pre-processing"
sys.path.append(path_train_test_split)
path_general_functions=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_functions)

from Train_Test_Split import training_set
from Train_Test_Split import test_set
from Renew_Set_EFM import renew_set_EFM
from TrainingAlgorithm_EFM import trainingalgorithm_EFM
from Reviews_Extraction import reviews_extraction
import numpy as np
import pandas as pd
import Ranking_Based_on_Relevance
import Get_UnbiasedDCG
import Get_UnbiasedRecall_EFM

class EFM(object):
    def __init__(self,df_train,df_test,A_train,A_test,X_train,X_test,Y_train,Y_test,expo_prob_test,N):
        self.df_train=df_train
        self.df_test=df_test
        self.A_train=A_train
        self.A_test=A_test
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.expo_prob_test=expo_prob_test
        self.N=N
        
    def performance_unbiasedNDCG(self,optimal_hp_combination):
         lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(optimal_hp_combination)
         r=int(r)
         r_=int(r_)
         k=int(k)
         U1,U2,V,H1,H2=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
         # compute X_,Y_,A_
         X_=U1.dot(V.T)
         Y_=U2.dot(V.T)
         A_=U1.dot(U2.T)+H1.dot(H2.T)
         df_test_predicted=renew_set_EFM(df_old=self.df_test,expo_prob=self.expo_prob_test,X=X_,Y=Y_,A=A_,N=self.N,k=k,c=c)
         df_test_actual=renew_set_EFM(df_old=self.df_test,expo_prob=self.expo_prob_test,X=self.X_test,Y=self.Y_test,A=self.A_test,N=self.N,k=k,c=c)
         # Assess the model's performance on unbiased DCG
         unbiased_DCG_sum_over_users=0
         unbiased_DCG_ideal_sum_over_users=0
         unique_user_indices_validation=np.unique(df_test_actual["user_index"].values).tolist()
         num_users=len(unique_user_indices_validation)
         for useri in unique_user_indices_validation:
             df_test_useri=df_test_actual[df_test_actual["user_index"]==useri]
             num_items=df_test_useri.shape[0]
             useri_list_relevance_predicted=list()
             useri_list_relevance_actual=list()
             useri_list_expo_prob=list()
             for useri_item_interaction in df_test_useri.itertuples():
                 user_index=useri
                 item_index=getattr(useri_item_interaction,"item_index")
                 expo_prob=getattr(useri_item_interaction,"exposure_probability")
                 relevance_actual=getattr(useri_item_interaction,"relevance")
                 useri_item_interactions=df_test_predicted[df_test_predicted["user_index"]==user_index]
                 relevance_predicted=useri_item_interactions[useri_item_interactions["item_index"]==item_index]["relevance"].values[0]
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
    def performance_unbiasedrecall(self,optimal_hp_combination):
        lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(optimal_hp_combination)
        r=int(r)
        r_=int(r_)
        k=int(k)
        U1,U2,V,H1,H2=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
        # compute X_,Y_,A_
        X_=U1.dot(V.T)
        Y_=U2.dot(V.T)
        A_=U1.dot(U2.T)+H1.dot(H2.T)
        df_test_predicted=renew_set_EFM(df_old=self.df_test,expo_prob=self.expo_prob_test,X=X_,Y=Y_,A=A_,N=self.N,k=k,c=c)
        df_test_actual=renew_set_EFM(df_old=self.df_test,expo_prob=self.expo_prob_test,X=self.X_test,Y=self.Y_test,A=self.A_test,N=self.N,k=k,c=c)
        # Assess the model's performance on unbiased Recall
        unbiased_recall_sum_over_users=0
        unique_user_indices_test=np.unique(df_test_actual["user_index"].values).tolist()
        num_users=len(unique_user_indices_test)
        for useri in unique_user_indices_test:
            threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
            unbiased_recall_useri=Get_UnbiasedRecall_EFM.get_unbiasedrecall_EFM(df_test_actual=df_test_actual,df_test_predicted=df_test_predicted,user_index=useri,threshold=threshold,k=k)
            unbiased_recall_sum_over_users+=unbiased_recall_useri
        unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
        return unbiased_recall_average_over_users