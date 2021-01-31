# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:26:17 2021

@author: Administrator
"""

"""
This script pre-processes the subject dataset to split it into a training set, a validation set, and a test set, finds the 
optimal hyperparameter combination through grid search cross validation, and trains the EFM model on
the training set and tests the trained model on the test set to assess the model's predictive accuracy via
unbiased DCG/recall.
"""
import sys
import os
import numpy as np
import pandas as pd

path_general_function="../../general functions"
sys.path.append(path_general_function)
from Reviews_Extraction import reviews_extraction
import Get_UnbiasedDCG
import Get_UnbiasedRecall

"""
Step 1: split the subject dataset into a training set, a validation set, and a test set.
"""
path_preprocessing="../../Data pre-processing"
sys.path.append(path_preprocessing)
import Train_Test_Split
path_dataset="../../../Datasets/Amazon_datasets/Phones_Accessories/Amazon_phones_accessories_N20.txt"
df=Train_Test_Split.txt_to_dataframe(path_dataset)
userids=df["userid"].values
itemids=df["itemid"].values
userid_to_index=Train_Test_Split.id_index_mapping(userids)
itemid_to_index=Train_Test_Split.id_index_mapping(itemids)
df=Train_Test_Split.replace_id_by_index(df=df, userid_to_index=userid_to_index, itemid_to_index=itemid_to_index)
#N=20 # n stands for the minimum number of interacted items required for a user to be in the denser n-core subset for train-test split.
#df_ncore=Train_Test_Split.Ncore(df,N)
df_ncore=df
m=len(np.unique(df_ncore["user_index"].values)) # the number of unique users
n=len(np.unique(df_ncore["item_index"].values)) # the number of unique items
ratings=list()
for interaction in df_ncore.itertuples():
    rating=getattr(interaction,"rating")
    if rating not in ratings:
        ratings.append(rating)
R_max=max(ratings)
test_size=10 
validation_size=5
df_train_validation=Train_Test_Split.training_set(df_ncore,m=test_size)
training_validation_indices=df_train_validation.index.tolist()
df_test=Train_Test_Split.test_set(df_ncore,training_validation_indices) 
df_train=Train_Test_Split.training_set(df_train_validation,m=validation_size)
training_indices=df_train.index.tolist()
df_validation=Train_Test_Split.test_set(df_train_validation,training_indices)
user_indices_train=np.unique(df_train["user_index"].values).tolist()
item_indices_train=np.unique(df_train["item_index"].values).tolist()
df_validation=Train_Test_Split.user_item_pair_check(user_indices_train,item_indices_train,df_validation)
df_test=Train_Test_Split.user_item_pair_check(user_indices_train,item_indices_train,df_test)

"""
Step 2: propensity scores' estimation for the training set and test set
"""
path_debiasing_tools="../../De-biasing tools"
sys.path.append(path_debiasing_tools)
import PropensityScores as PS
propensityscores_train=PS.PropensityScores(df_train)
propensityscores_validation=PS.PropensityScores(df_validation)
propensityscores_test=PS.PropensityScores(df_test)
power=0.5 # the power of the power-law distribution for the user-independent approach to propensity score estimations is set as 0.5
propensityscores_train.user_independent_PS(power)
propensityscores_validation.user_independent_PS(power)
propensityscores_test.user_independent_PS(power)
expo_prob_train=propensityscores_train.expo_prob
expo_prob_validation=propensityscores_validation.expo_prob
expo_prob_test=propensityscores_test.expo_prob

"""
 Step 3: Extract subsets from reviews for validation and testing respectively 
"""
path_opinion_matrix="../../Opinion Matrix"
sys.path.append(path_opinion_matrix)
import Replace_ID_with_Index
import pickle
path_reviews="../../../Datasets/Amazon_datasets/Phones_Accessories/Amazon_phones_accessories_N20_quadruples/reviews.pickle"
reviews=pickle.load(open(path_reviews,"rb"))
reviews=Replace_ID_with_Index.replace_id_with_index(reviews=reviews,userid_to_index=userid_to_index,itemid_to_index=itemid_to_index)
user_item_pairs_train=list()
user_item_pairs_validation=list()
user_item_pairs_test=list()
for interaction in df_train.itertuples():
    user_item_pair=(getattr(interaction,"user_index"),getattr(interaction,"item_index"))
    user_item_pairs_train.append(user_item_pair)
for interaction in df_validation.itertuples():
    user_item_pair=(getattr(interaction,"user_index"),getattr(interaction,"item_index"))
    user_item_pairs_validation.append(user_item_pair)
for interaction in df_test.itertuples():
    user_item_pair=(getattr(interaction,"user_index"),getattr(interaction,"item_index"))
    user_item_pairs_test.append(user_item_pair)
reviews_train=reviews_extraction(user_item_pairs_train,reviews)
reviews_validation=reviews_extraction(user_item_pairs_validation,reviews)
reviews_test=reviews_extraction(user_item_pairs_test,reviews)
reviews_train_validation=reviews_train+reviews_validation


"""
Step 4: Construct the user-item rating matrix (A), user-feature attention matrix (X), and item-feature quality matrix (Y) based on the reviews for validation and reviews for testing.  
"""
path_EFM="./"
sys.path.append(path_EFM)
import Get_Matrices_EFM
unique_features=list()
for review in reviews:
    if "sentence" not in review.keys():continue
    feature=review["sentence"][0][0]
    if feature not in unique_features:
        unique_features.append(feature)
p=len(unique_features)
get_matrices_train=Get_Matrices_EFM.get_matrices(reviews_train,m,n,p,N=R_max)
get_matrices_validation=Get_Matrices_EFM.get_matrices(reviews_validation,m,n,p,R_max)
get_matrices_train_validation=Get_Matrices_EFM.get_matrices(reviews_train_validation,m,n,p,N=R_max)
get_matrices_test=Get_Matrices_EFM.get_matrices(reviews_test,m,n,p,N=R_max)
get_matrices_train.user_item_rating_matrix()
get_matrices_validation.user_item_rating_matrix()
get_matrices_train.user_feature_attention_matrix()
get_matrices_train.item_feature_quality_matrix()
get_matrices_train_validation.user_item_rating_matrix()
get_matrices_test.user_item_rating_matrix()
get_matrices_train_validation.user_feature_attention_matrix()
get_matrices_train_validation.item_feature_quality_matrix()
A_train=get_matrices_train.A
A_validation=get_matrices_validation.A
X_train=get_matrices_train.X
Y_train=get_matrices_train.Y
A_train_validation=get_matrices_train_validation.A
A_test=get_matrices_test.A
X_train_validation=get_matrices_train_validation.X
Y_train_validation=get_matrices_train_validation.Y
"""
Step 6: Optimal hyperparameter search via grid search cross validation 
"""

import Hyperparameters as hp
import HyperparameterTuning_Train_Validation_Split_EFM as ht_trainvalidation

#param_grid={
    #"lambda_x":[0.1,1e-5],
    #"lambda_y":[0.1,1e-5],
    #"lambda_u":[0.1,1e-5],
    #"lambda_h":[0.1,1e-5],
    #"lambda_v":[0.1,1e-5],
    #"r":[20,80],
    #"r_":[80,20],
    #"c":[0.2,0.8],
    #"k":[5,int(np.floor((p-5)/5+1))]}
param_grid={
    "lambda_x":[0.1,0.01,0.001,1e-4,1e-5],
    "lambda_y":[0.1,0.01,0.001,1e-4,1e-5],
    "lambda_u":[0.1,0.01,0.001,1e-4,1e-5],
    "lambda_h":[0.1,0.01,0.001,1e-4,1e-5],
    "lambda_v":[0.1,0.01,0.001,1e-4,1e-5],
    "r":[0,20,40,60,80,100],
    "r_":[100,80,60,40,20,0],
    "c":[0,0.2,0.5,0.8,1],
    "k":[int(k) for k in np.linspace(start=5,stop=p,num=int(np.floor((p-5)/5+1)),endpoint=True)]}
hp_combinations=hp.hyperparameter_combinations(param_grid)

hp_train_validation_split=ht_trainvalidation.hyperparametertuning_train_validation_split_EFM(df_train,df_validation,A_train,A_validation,X_train,Y_train,expo_prob_validation,hp_combinations)
#hp_train_validation_split.train_validation_split_unbiasedMSE() # hyperparameter tuning via unbiased RMSE
hp_train_validation_split.train_validation_split_unbiasedDCG() # hyperparameter tuning via unbiased NDCG
#k_validation=3 # test the model's performance on the validation set through top-k_validation recommendation
#hp_train_validation_split.train_validation_split_unbiasedrecall(k=k_validation) # hyperparameter tuning via unbiased recall

# find the optimal hyperparameter combination evaluated on unbiased RMSE
#optimal_hp_combination_unbiasedMSE=hp_train_validation_split.hp_combinations[np.argmin(hp_train_validation_split.hp_combination_performance_unbiasedMSE)]

# find the optimal hyperparameter combination evaluated on unbiased NDCG
optimal_hp_combination_unbiasedDCG=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedDCG)]

#find the optimal hyperparameter combination evaluated on unbiased recall
#optimal_hp_combination_unbiasedrecall=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedrecall)]

"""
Step 5: USMF training, testing and performance evaluation for top-K recommendation
"""
df_train=pd.concat([df_train,df_validation],axis=0)
expo_prob_test=expo_prob_test
from EFM import EFM
efm=EFM(df_train,df_test,A_train_validation,X_train_validation,Y_train_validation,expo_prob_test)


# compute the model's performance on the test set evluated via unbiased RMSE
#unbiased_MSE=efm.performance_unbiasedMSE(optimal_hp_combination_unbiasedMSE)
#print("The model's performance evaluated via unbiased MSE is {}".format(unbiased_MSE))

# compute the model's performance on the test set evluated via unbiased NDCG
unbiased_DCG=efm.performance_unbiasedDCG(optimal_hp_combination_unbiasedDCG)
print("The model's performance evaluated via unbiased DCG is {}".format(unbiased_DCG))

# compute the model's performance on the test set evaluated via unbiased recall
#k_test=5 # test the model's performance on the test set through top-k_test recommendation
#unbiased_recall=efm.performance_unbiasedrecall(optimal_hp_combination_unbiasedrecall,k=k_test)
#print("The model's performance evaluated via unbiased recall is {}".format(unbiased_recall))
