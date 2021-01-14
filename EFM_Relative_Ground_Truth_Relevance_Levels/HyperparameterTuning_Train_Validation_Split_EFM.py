# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:13:50 2020

@author: Administrator
"""
import os
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
from Checking_Convergence import checking_convergence


class hyperparametertuning_Train_Validation_Split_EFM(object):
    def __init__(self,df_train,df_validation,A_train,A_validation,X_train,X_validation,Y_train,Y_validation,expo_prob_train,expo_prob_validation,N,hp_combinations):
        """
        Parameters
        ----------
        df: the dataset stored as a Pandas dataframe.
        A: the user-item rating matrix.
        X: the user-feature attention matrix.
        Y: the item-feature quality matrix.
        expo_prob: the estimates for exposure probabilities
        N: the greatest possible value on the numerical rating scale.
        Returns
        -------
        None.
        """
        self.df_train=df_train
        self.df_validation=df_validation
        self.A_train=A_train
        self.A_validation=A_validation
        self.X_train=X_train
        self.X_validation=X_validation
        self.Y_train=Y_train
        self.Y_validation=Y_validation
        self.expo_prob_train=expo_prob_train
        self.expo_prob_validation=expo_prob_validation
        self.N=N
        self.hp_combinations=hp_combinations
        self.hp_combination_performance_unbiasedNDCG=list()
        self.hp_combination_performance_unbiasedrecall=list()
    #def result_recording(file_path,metric,hp_combination,performance):
        #f=open(file_path,"w",encoding="utf-8") 
        #f.write(str([metric,hp_combination,performance])+"\n")
        #f.close()
    def train_validation_split_unbiasedNDCG(self):
        base_path=r"C:\Users\Administrator\Desktop\上科大\代码\Baselines\EFM"
        file_name="EFM_Validation_Results_unbiasedNDCG.txt"
        result_path=os.path.join(base_path,file_name)
        f=open(result_path,"w",encoding="utf-8")
        for index,hp_combination in enumerate(self.hp_combinations): # each hp_combination is of the following form:[lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k]
            lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(hp_combination)
            r=int(r)
            r_=int(r_)
            k=int(k)
            # Variable "matrices" has the following form:[U1,U2,V,H1,H2]
            matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
            if index==0:
                checking_convergence(iterations=matrices[5],costs=matrices[6])
            U1=matrices[0]
            U2=matrices[1]
            V=matrices[2]
            H1=matrices[3]
            H2=matrices[4]
            # compute X_,Y_,A_
            X_=U1.dot(V.T)
            Y_=U2.dot(V.T)
            A_=U1.dot(U2.T)+H1.dot(H2.T)
            df_validation_predicted=renew_set_EFM(df_old=self.df_validation,expo_prob=self.expo_prob_validation,X=X_,Y=Y_,A=A_,N=self.N,k=k,c=c)
            df_validation_actual=renew_set_EFM(df_old=self.df_validation,expo_prob=self.expo_prob_validation,X=self.X_validation,Y=self.Y_validation,A=self.A_validation,N=self.N,k=k,c=c)
            # Assess the model's performance on unbiased DCG
            unbiased_DCG_sum_over_users=0
            unbiased_DCG_ideal_sum_over_users=0
            unique_user_indices_validation=np.unique(df_validation_actual["user_index"].values).tolist()
            num_users=len(unique_user_indices_validation)
            for useri in unique_user_indices_validation:
                df_validation_useri=df_validation_actual[df_validation_actual["user_index"]==useri]
                num_items=df_validation_useri.shape[0]
                useri_list_relevance_predicted=list()
                useri_list_relevance_actual=list()
                useri_list_expo_prob=list()
                for useri_item_interaction in df_validation_useri.itertuples():
                    user_index=useri
                    item_index=getattr(useri_item_interaction,"item_index")
                    expo_prob=getattr(useri_item_interaction,"exposure_probability")
                    relevance_actual=getattr(useri_item_interaction,"relevance")
                    useri_item_interactions=df_validation_predicted[df_validation_predicted["user_index"]==user_index]
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
            f.write(str([hp_combination,"unbiased NDCG",unbiased_NDCG])+"\n")
            self.hp_combination_performance_unbiasedNDCG.append(unbiased_NDCG)
        f.close()
    def train_validation_split_unbiasedrecall(self,k):
        base_path=r"C:\Users\Administrator\Desktop\上科大\代码\Baselines\EFM"
        file_name="EFM_Validation_Results_unbiasedrecall.txt"
        result_path=os.path.join(base_path,file_name)
        f=open(result_path,"w",encoding="utf-8")
        for index,hp_combination in enumerate(self.hp_combinations):
            lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(hp_combination)
            r=int(r)
            r_=int(r_)
            k=int(k)
            # Variable "matrices" has the following form:[U1,U2,V,H1,H2]
            matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
            U1=matrices[0]
            U2=matrices[1]
            V=matrices[2]
            H1=matrices[3]
            H2=matrices[4]
            # compute X_,Y_,A_
            X_=U1.dot(V.T)
            Y_=U2.dot(V.T)
            A_=U1.dot(U2.T)+H1.dot(H2.T)
            df_validation_predicted=renew_set_EFM(df_old=self.df_validation,expo_prob=self.expo_prob_validation,X=X_,Y=Y_,A=A_,N=self.N,k=k,c=c)
            df_validation_actual=renew_set_EFM(df_old=self.df_validation,expo_prob=self.expo_prob_validation,X=self.X_validation,Y=self.Y_validation,A=self.A_validation,N=self.N,k=k,c=c)
            # Assess the model's performance on unbiased Recall
            unbiased_recall_sum_over_users=0
            unique_user_indices_test=np.unique(df_validation_actual["user_index"].values).tolist()
            num_users=len(unique_user_indices_test)
            for useri in unique_user_indices_test:
                threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
                unbiased_recall_useri=Get_UnbiasedRecall_EFM.get_unbiasedrecall_EFM(df_test_actual=df_validation_actual,df_test_predicted=df_validation_predicted,user_index=useri,threshold=threshold,k=k)
                unbiased_recall_sum_over_users+=unbiased_recall_useri
            unbiased_recall_average_over_users=unbiased_recall_sum_over_users/num_users
            f.write(str([hp_combination,"unbiased recall",unbiased_recall_average_over_users])+"\n")
            self.hp_combination_performance_unbiasedrecall.append(unbiased_recall_average_over_users)
        f.close()