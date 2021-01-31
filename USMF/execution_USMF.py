# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:29:57 2021

@author: Administrator
"""

import sys
import os
import numpy as np
import pandas as pd


"""
This script pre-processes the subject dataset to split it into a training set and a test set, finds the 
optimal hyperparameter combination through grid search cross validation, and trains the USMF model on
the training set and tests the trained model on the test set to assess the model's predictive accuracy via
unbiased RMSE, unbiased NDCG, and unbiased recall.
"""

path_general_function="../General functions"
sys.path.append(path_general_function)
from Reviews_Extraction import reviews_extraction
import Get_UnbiasedDCG
import Get_UnbiasedRecall

"""
Step 1: split the subject dataset into a training set, a validation set, and a test set.
"""
path_preprocessing="../pre-processing"
sys.path.append(path_preprocessing)
import Train_Test_Split
path_dataset="../../Datasets/Amazon_datasets/Phones_Accessories/Amazon_phones_accessories_N20.txt"
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
path_debiasing_tools="../De-biasing tools"
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
path_opinion_matrix=r"C:\Users\Administrator\Desktop\上科大\代码\Opinion Matrix"
sys.path.append(path_opinion_matrix)
import Get_Matrices, Replace_ID_with_Index
import pickle
path_reviews="../../Datasets/Amazon_datasets/Phones_Accessories/Amazon_phones_accessories_N20_quadruples/reviews.pickle"
reviews=pickle.load(open(path_reviews,"rb"))
reviews=Replace_ID_with_Index.replace_id_with_index(reviews=reviews,userid_to_index=userid_to_index,itemid_to_index=itemid_to_index)
user_item_pairs_train=list()
user_item_pairs_validation=list()
for interaction in df_train.itertuples():
    user_item_pair=(getattr(interaction,"user_index"),getattr(interaction,"item_index"))
    user_item_pairs_train.append(user_item_pair)
for interaction in df_validation.itertuples():
    user_item_pair=(getattr(interaction,"user_index"),getattr(interaction,"item_index"))
    user_item_pairs_validation.append(user_item_pair)
user_item_pairs_train=user_item_pairs_train
user_item_pairs_train_validation=user_item_pairs_train+user_item_pairs_validation
reviews_validation=reviews_extraction(user_item_pairs_train, reviews)
reviews_test=reviews_extraction(user_item_pairs_train_validation,reviews)

"""
Step 4: Construct the user-feature attention matrix (X) and item-feature quality matrix (Y) based on the reviews for validation and reviews for testing.  
"""
unique_features=list()
for review in reviews:
    if "sentence" not in review.keys():continue
    feature=review["sentence"][0][0]
    if feature not in unique_features:
        unique_features.append(feature)
p=len(unique_features)
get_matrices_validation=Get_Matrices.get_matrices(reviews=reviews_validation,m=m,n=n,p=p,N=R_max)
get_matrices_test=Get_Matrices.get_matrices(reviews=reviews_test,m=m,n=n,p=p,N=R_max)
get_matrices_validation.user_feature_attention_matrix()
get_matrices_validation.item_feature_quality_matrix()
get_matrices_test.user_feature_attention_matrix()
get_matrices_test.item_feature_quality_matrix()
X_train=get_matrices_validation.X
Y_train=get_matrices_validation.Y
X_train_validation=get_matrices_test.X
Y_train_validation=get_matrices_test.Y


"""
Step 5: Optimal hyperparameter search via grid search cross validation
"""
"""
For each hp_combination stored as a two-dimensional numpy array, 
the first entry is a choice for the regularization coefficient (lambda),
the second entry is a choice for the number of latent factors (k),
the third entry is a choice for the rating-opinion relative contribution weight (c)
"""
path_USMF_algorithm_training="./"
sys.path.append(path_USMF_algorithm_training)
import Hyperparameters as hp
import HyperparameterTuning_TrainValidationSplit_USMF as ht_trainvalidation
param_grid={
            "regularization coefficient_rating_factors":[0.1,0.01,0.001,1e-4,1e-5],
            "regularization coefficient_biases":[0.1,0.01,0.001,1e-4,1e-5],
            "rating-opinion relative contribution weight":[0,0.2,0.5,0.7,1]
            }
hp_combinations=hp.hyperparameter_combinations(param_grid)
hp_train_validation_split=ht_trainvalidation.hyperparametertuning_train_validation_split_USMF(df_train,df_validation,expo_prob_train,expo_prob_validation,X_train,Y_train,hp_combinations)
#hp_train_validation_split.train_validation_split_unbiasedMSE() # hyperparameter tuning via unbiased RMSE
hp_train_validation_split.train_validation_split_unbiasedDCG() # hyperparameter tuning via unbiased DCG
k_validation=3 # test the model's performance on the validation set through top-k_validation recommendation
hp_train_validation_split.train_validation_split_unbiasedrecall(k=k_validation) # hyperparameter tuning via unbiased recall

# find the optimal hyperparameter combination evaluated on unbiased RMSE
#optimal_hp_combination_unbiasedMSE=hp_train_validation_split.hp_combinations[np.argmin(hp_train_validation_split.hp_combination_performance_unbiasedMSE)]

# find the optimal hyperparameter combination evaluated on unbiased NDCG
#optimal_hp_combination_unbiasedDCG=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedDCG)]

#find the optimal hyperparameter combination evaluated on unbiased recall @3
optimal_hp_combination_unbiasedrecall=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedrecall)]

"""
Step 5: USMF training, testing and performance evaluation for top-K recommendation
"""
df_train=pd.concat([df_train,df_validation],axis=0)
expo_prob_train=expo_prob_train+expo_prob_validation
expo_prob_test=expo_prob_test
from USMF import USMF
usmf=USMF(df_train,df_test,expo_prob_train,expo_prob_test,X_train_validation,Y_train_validation)


# compute the model's performance on the test set evluated via unbiased RMSE
#unbiased_MSE=usmf.performance_unbiasedRMSE(optimal_hp_combination_unbiasedMSE)
#print("The model's performance evaluated via unbiased MSE is {}".format(unbiased_MSE))

# compute the model's performance on the test set evluated via unbiased NDCG
#unbiased_DCG=usmf.performance_unbiasedDCG(optimal_hp_combination_unbiasedDCG)
#print("The model's performance evaluated via unbiased DCG is {}".format(unbiased_DCG))

# compute the model's performance on the test set evaluated via unbiased recall
k_test=5 # test the model's performance on the test set through top-k_test recommendation
unbiased_recall=usmf.performance_unbiasedrecall(optimal_hp_combination_unbiasedrecall,k=k_test)
print("The model's performance evaluated via unbiased recall @{0} is {1}".format(k_test,unbiased_recall))