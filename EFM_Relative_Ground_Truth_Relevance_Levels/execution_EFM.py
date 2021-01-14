# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:26:17 2021

@author: Administrator
"""

"""
This script pre-processes the subject dataset to split it into a training set, a validation set, and a test set, finds the 
optimal hyperparameter combination through grid search cross validation, and trains the EFM model on
the training set and tests the trained model on the test set to assess the model's ranking performance via
unbiased DCG/recall.
"""
import sys
import os
import numpy as np
import pandas as pd

path_general_function=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_function)
from Reviews_Extraction import reviews_extraction

"""
Step 1: split the subject dataset into a training set, a validation set, and a test set.
"""
path_preprocessing=r"C:\Users\Administrator\Desktop\上科大\代码\Data pre-processing"
sys.path.append(path_preprocessing)
import Train_Test_Split
path_dataset=r"C:\Users\Administrator\Desktop\上科大\数据集\Yelp_dataset\Las_Vegas\Food_Restaurant_Reviews_Las_Vegas_20core_Yelp.txt"
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
test_size=5  
df_train_validation=Train_Test_Split.training_set(df_ncore,m=test_size)
training_validation_indices=df_train_validation.index.tolist()
df_test=Train_Test_Split.test_set(df_ncore,training_validation_indices)
validation_size=5   
df_train=Train_Test_Split.training_set(df_train_validation,m=validation_size)
training_indices=df_train.index.tolist()
df_validation=Train_Test_Split.test_set(df_train_validation,training_indices)

#
"""
Step 2: Split the reviews into a subset for training, a subset for validation, and a subset for testing respectively. 
"""
path_Opinion_Matrix=r"C:\Users\Administrator\Desktop\上科大\代码\Opinion Matrix"
sys.path.append(path_Opinion_Matrix)
import Replace_ID_with_Index
import pickle
path_reviews=r"C:\Users\Administrator\Desktop\English-Jar\lei\output\Las Vegas_20core\reviews.pickle"
reviews=pickle.load(open(path_reviews,"rb"))
reviews=Replace_ID_with_Index.replace_id_with_index(reviews=reviews,userid_to_index=userid_to_index,itemid_to_index=itemid_to_index)
unique_features=list()
for review in reviews:
    if "sentence" in review.keys():
        if review["sentence"][0][0] not in unique_features:
            unique_features.append(review["sentence"][0][0])
p=len(unique_features) # the number of unique product features
rating_scale=list()
for review in reviews:
    if review["rating"] not in rating_scale:
        rating_scale.append(float(review["rating"]))
R_max=max(rating_scale) 

user_item_pairs_train=list()
user_item_pairs_validation=list()
user_item_pairs_test=list()
for interaction in df_train.itertuples():
    user=getattr(interaction,"user_index")
    item=getattr(interaction,"item_index")
    user_item_pair=(user,item)
    user_item_pairs_train.append(user_item_pair)
for interaction in df_validation.itertuples():
    user=getattr(interaction,"user_index")
    item=getattr(interaction,"item_index")
    user_item_pair=(user,item)
    user_item_pairs_validation.append(user_item_pair)
for interaction in df_test.itertuples():
    user=getattr(interaction,"user_index")
    item=getattr(interaction,"item_index")
    user_item_pair=(user,item)
    user_item_pairs_test.append(user_item_pair)
reviews_train=reviews_extraction(user_item_pairs_train,reviews)
reviews_validation=reviews_extraction(user_item_pairs_validation,reviews)
reviews_test=reviews_extraction(user_item_pairs_test,reviews)
reviews_train=reviews_train
reviews_validation=reviews_train+reviews_validation
reivews_test=reviews_validation+reviews_test

"""
Step 4: Compute the user-item rating matrix (A), user-feature attention matrix (X), and item-feature quality matrix (Y) for training, validation, and testing respectively. 
"""
path_get_matrices=r"C:\Users\Administrator\Desktop\上科大\代码\Baselines\EFM"
sys.path.append(path_get_matrices)
from Get_Matrices import get_matrices
matrices_obj_train=get_matrices(reviews=reviews_train,m=m,n=n,p=p,N=R_max)
matrices_obj_validation=get_matrices(reviews=reviews_validation,m=m,n=n,p=p,N=R_max)
matrices_obj_test=get_matrices(reviews=reviews_test,m=m,n=n,p=p,N=R_max)
matrices_obj_train.user_item_rating_matrix()
matrices_obj_train.user_feature_attention_matrix()
matrices_obj_train.item_feature_quality_matrix()
A_train=matrices_obj_train.A
X_train=matrices_obj_train.X
Y_train=matrices_obj_train.Y
matrices_obj_validation.user_item_rating_matrix()
matrices_obj_validation.user_feature_attention_matrix()
matrices_obj_validation.item_feature_quality_matrix()
A_validation=matrices_obj_validation.A
X_validation=matrices_obj_validation.X
Y_validation=matrices_obj_validation.Y
matrices_obj_test.user_item_rating_matrix()
matrices_obj_test.user_feature_attention_matrix()
matrices_obj_test.item_feature_quality_matrix()
A_test=matrices_obj_test.A
X_test=matrices_obj_test.X
Y_test=matrices_obj_test.Y

"""
Step 5: propensity scores' estimation for the training set and test set
"""
path_debiasing_tools=r"C:\Users\Administrator\Desktop\上科大\代码\De-biasing tools"
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
Step 6: Optimal hyperparameter search via grid search cross validation 
"""
import Hyperparameters as hp
import HyperparameterTuning_Train_Validation_Split_EFM as ht_trainvalidation


param_grid={
    "lambda_x":[0.1,1e-5],
    "lambda_y":[0.1,1e-5],
    "lambda_u":[0.1,1e-5],
    "lambda_h":[0.1,1e-5],
    "lambda_v":[0.1,1e-5],
    "r":[20,80],
    "r_":[80,20],
    "c":[0.2,0.8],
    "k":[5,int(np.floor((p-5)/5+1))]}
#param_grid={
    #"lambda_x":[0.1,0.01,0.001,1e-4,1e-5],
    #"lambda_y":[0.1,0.01,0.001,1e-4,1e-5],
    #"lambda_u":[0.1,0.01,0.001,1e-4,1e-5],
    #"lambda_h":[0.1,0.01,0.001,1e-4,1e-5],
    #"lambda_v":[0.1,0.01,0.001,1e-4,1e-5],
    #"r":[0,20,40,60,80,100],
    #"r_":[100,80,60,40,20,0],
    #"c":[0,0.2,0.5,0.8,1],
    #"k":[int(k) for k in np.linspace(start=5,stop=p,num=int(np.floor((p-5)/5+1)),endpoint=True)]}
hp_combinations=hp.hyperparameter_combinations(param_grid)
ht_gscv=ht_trainvalidation.hyperparametertuning_Train_Validation_Split_EFM(df_train=df_train, df_validation=df_validation, A_train=A_train, A_validation=A_validation, X_train=X_train, X_validation=X_validation, Y_train=Y_train, Y_validation=Y_validation, expo_prob_train=expo_prob_train, expo_prob_validation=expo_prob_validation,N=R_max,hp_combinations=hp_combinations)
ht_gscv.train_validation_split_unbiasedNDCG() # hyperparameter tuning via unbiased NDCG
#k_validation=3 # the number of most highly ranked items for each user for the purpose of computing unbiased recall for hyperparameter tuning
#ht_gscv.train_validation_split_unbiasedrecall() # hyperparameter tuning via unbiased recall

# find the optimal hyperparameter combination evaluated on unbiased NDCG
optimal_hp_combination_unbiasedNDCG=ht_gscv.hp_combinations[np.argmax(ht_gscv.hp_combination_performance_unbiasedNDCG)]

# find the optimal hyperparameter combination evaluated on unbiased recall
#optimal_hp_combination_unbiasedrecall=ht_gscv.hp_combinations[np.argmax(ht_gscv.hp_combination_performance_unbiasedrecall)]

"""
Step 7: Train the EFM model on the training set and assess the trained model's performance on the test set through unbiased NDCG and unbiased recall. 
"""
df_train=pd.concat([df_train,df_validation],axis=0)
expo_prob_train=expo_prob_train+expo_prob_validation
expo_prob_test=expo_prob_test
A_train=A_train+A_validation
X_train=X_train+X_validation
Y_train=Y_train+Y_validation
from EFM import EFM
# compute the model's performance on the test set evluated via unbiased NDCG
efm=EFM(df_train=df_train,df_test=df_test,A_train=A_train,A_test=A_test,X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test,expo_prob_test=expo_prob_test,N=R_max)
unbiasedNDCG=efm.performance_unbiasedNDCG(optimal_hp_combination_unbiasedNDCG)
print("The model's performance evaluated via unbiased NDCG is {}".format(unbiasedNDCG))

# compute the model's performance on the test set evaluated via unbiased recall
#k_test=3 # test the model's performance on the test set through top-k_test recommendation
#unbiasedrecall=efm.performance_unbiasedrecall(optimal_hp_combination_unbiasedrecall)
#print("The model's performance evaluated via unbiased recall is {}".format(unbiasedrecall)))
