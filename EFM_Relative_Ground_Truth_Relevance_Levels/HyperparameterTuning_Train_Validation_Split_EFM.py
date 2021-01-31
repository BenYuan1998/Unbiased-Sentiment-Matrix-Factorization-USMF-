# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:13:50 2020

@author: Administrator
"""
import os
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
from Checking_Convergence import checking_convergence


class hyperparametertuning_train_validation_split_EFM(object):
    def __init__(self,df_train,df_validation,A_train,A_validation,X_train,Y_train,expo_prob_validation,hp_combinations):
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
        self.Y_train=Y_train
        self.expo_prob_validation=expo_prob_validation
        self.hp_combinations=hp_combinations
        self.hp_combination_performance_unbiasedMSE=list()
        self.hp_combination_performance_unbiasedDCG=list()
        self.hp_combination_performance_unbiasedrecall=list()
    #def result_recording(file_path,metric,hp_combination,performance):
        #f=open(file_path,"w",encoding="utf-8") 
        #f.write(str([metric,hp_combination,performance])+"\n")
        #f.close()
    def train_validation_split_unbiasedMSE(self):
        for index,hp_combination in enumerate(self.hp_combinations): # each hp_combination is of the following form:[lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k]
            lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(hp_combination)
            r=int(r)
            r_=int(r_)
            k=int(k)
            # Variable "matrices" has the following form:[U1,U2,V,H1,H2]
            matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
            if index==0:
                iterations=matrices[5]
                costs=matrices[6]
                checking_convergence(iterations,costs)
            U1=matrices[0]
            U2=matrices[1]
            H1=matrices[3]
            H2=matrices[4]
            # compute A_
            A_=U1.dot(U2.T)+H1.dot(H2.T)
            # Assess the model's performance on unbiased MSE
            MSE_sum_over_items=0
            unique_item_indices_validation=np.unique(self.df_validation["item_index"].values).tolist()
            num_items=len(unique_item_indices_validation)
            for itemi in unique_item_indices_validation:
                SE_itemi=0
                df_validation_itemi=self.df_validation[self.df_validation["item_index"]==itemi]
                num_interactions=df_validation_itemi.shape[0]
                for interaction in df_validation_itemi.itertuples():
                    user_index=getattr(interaction,"user_index")
                    item_index=itemi
                    relevance_predicted=A_[user_index,item_index]
                    relevance_actual=self.A_validation[user_index,item_index]
                    square_error=(relevance_predicted-relevance_actual)**2
                    SE_itemi+=square_error
                MSE_itemi=SE_itemi/num_interactions
                MSE_sum_over_items+=MSE_itemi
            unbiased_MSE=MSE_sum_over_items/num_items
            self.hp_combination_performance_unbiasedMSE.append(unbiased_MSE)
            if index==0:
                print("Validation ends for the 1st hyperparameter combination.")
            elif index==1:
                print("Validation ends for the 2nd hyperparameter combination.")
            elif index==2:
                print("Validation ends for the 3rd hyperparameter combination.")
            else:
                print("Validation ends for the {}th hyperparameter combination".format(index+1))
    def train_validation_split_unbiasedDCG(self):
        for index,hp_combination in enumerate(self.hp_combinations): # each hp_combination is of the following form:[lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k]
            lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(hp_combination)
            r=int(r)
            r_=int(r_)
            k=int(k)
            # Variable "matrices" has the following form:[U1,U2,V,H1,H2]
            df_validation=renew_set(self.df_validation,self.expo_prob_validation)
            matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
            U1=matrices[0]
            U2=matrices[1]
            H1=matrices[3]
            H2=matrices[4]
            # compute A_
            A_=U1.dot(U2.T)+H1.dot(H2.T)
            # Assess the model's performance on unbiased DCG
            unbiased_DCG_sum_over_users=0
            unique_user_indices_validation=np.unique(df_validation["user_index"].values).tolist()
            num_users=len(unique_user_indices_validation)
            for useri in unique_user_indices_validation:
                threshold=3
                unbiased_DCG_useri=Get_UnbiasedDCG_EFM.get_unbiasedDCG_per_user(df_validation, useri, A_, threshold)
                unbiased_DCG_sum_over_users+=unbiased_DCG_useri
            unbiased_DCG=unbiased_DCG_sum_over_users/num_users
            self.hp_combination_performance_unbiasedDCG.append(unbiased_DCG)
    def train_validation_split_unbiasedrecall(self,k):
        for index,hp_combination in enumerate(self.hp_combinations):
            lambda_x,lambda_y,lambda_u,lambda_h,lambda_v,r,r_,c,k=list(hp_combination)
            r=int(r)
            r_=int(r_)
            k=int(k)
            df_validation=renew_set(self.df_validation,self.expo_prob_validation)
            # Variable "matrices" has the following form:[U1,U2,V,H1,H2]
            matrices=trainingalgorithm_EFM(A=self.A_train,X=self.X_train,Y=self.Y_train,r=r,r_=r_,lambda_x=lambda_x,lambda_y=lambda_y,lambda_u=lambda_u,lambda_h=lambda_h,lambda_v=lambda_v)
            U1=matrices[0]
            U2=matrices[1]
            H1=matrices[3]
            H2=matrices[4]
            # compute A_
            A_=U1.dot(U2.T)+H1.dot(H2.T)
            # Assess the model's performance on unbiased Recall
            unbiased_recall_sum_over_users=0
            unique_user_indices_validation=np.unique(df_validation["user_index"].values).tolist()
            num_users=len(unique_user_indices_validation)
            for useri in unique_user_indices_validation:
                threshold=3 # The cut-off relevance level for categorizing items in the test set into the set of relevant items and the set of irrelevant items is set as 3.
                unbiased_recall_useri=Get_UnbiasedRecall_EFM.get_unbiasedrecall_EFM(df_validation, useri, A_, threshold,k)
                unbiased_recall_sum_over_users+=unbiased_recall_useri
            unbiased_recall=unbiased_recall_sum_over_users/num_users
            self.hp_combination_performance_unbiasedrecall.append(unbiased_recall)
    