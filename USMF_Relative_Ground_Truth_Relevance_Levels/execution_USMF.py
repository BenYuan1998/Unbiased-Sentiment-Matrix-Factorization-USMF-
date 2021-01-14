# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:43:45 2020

@author: Administrator
"""
import sys
import os
import numpy as np
import pandas as pd


"""
This script pre-processes the subject dataset to split it into a training set and a test set, finds the 
optimal hyperparameter combination through grid search cross validation, and trains the USMF model on
the training set and tests the trained model on the test set to assess the model's ranking performance via
unbiased DCG/recall.
"""
path_general_function=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_function)

"""
Step 1: Construct the id-index mappings for both users and items. 
"""
path_preprocessing=r"C:\Users\Administrator\Desktop\上科大\代码\Data pre-processing"
sys.path.append(path_preprocessing)
import Train_Test_Split
path_dataset=r"C:\Users\Administrator\Desktop\上科大\数据集\Yelp_dataset\Las_Vegas_Food_Restaurant\Food_Restaurant_Reviews_Las_Vegas_20core_Yelp.txt"
df=Train_Test_Split.txt_to_dataframe(path_dataset)
userids=df["userid"].values
itemids=df["itemid"].values
userid_to_index=Train_Test_Split.id_index_mapping(userids)
itemid_to_index=Train_Test_Split.id_index_mapping(itemids)
df=Train_Test_Split.replace_id_by_index(df=df, userid_to_index=userid_to_index, itemid_to_index=itemid_to_index)
df_ncore=df
m=len(np.unique(df_ncore["user_index"].values)) # the number of unique users
n=len(np.unique(df_ncore["item_index"].values)) # the number of unique items
ratings=list()
for interaction in df_ncore.itertuples():
    rating=getattr(interaction,"rating")
    if rating not in ratings:
        ratings.append(rating)
R_max=max(ratings)

"""
Step 2: Load the training set, validation set, and the test set.
"""
path_Las_Vegas_food_restaurant=r"C:\Users\Administrator\Desktop\上科大\数据集\Yelp_dataset\Las_Vegas_Food_Restaurant"
df_train_file_name="Training Set.csv"
df_validation_file_name="Validation Set.csv"
df_test_file_name="Test Set.csv"
df_train=pd.read_csv(os.path.join(path_Las_Vegas_food_restaurant,df_train_file_name))
df_validation=pd.read_csv(os.path.join(path_Las_Vegas_food_restaurant,df_validation_file_name))
df_test=pd.read_csv(os.path.join(path_Las_Vegas_food_restaurant,df_test_file_name))
df_validation.drop(columns=["feature_based_score"],inplace=True)
df_test.drop(columns=["feature_based_score"],inplace=True)

"""
Step 3: propensity scores' estimation for the training set and test set
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
 Step 3: Compute the user-feature attention matrix (X) and the item-feature quality matrix (Y) based on the entire sentiment lexicon.
"""
path_opinion_matrix=r"C:\Users\Administrator\Desktop\上科大\代码\Opinion Matrix"
sys.path.append(path_opinion_matrix)
import Get_Matrices, Replace_ID_with_Index
import pickle
path_reviews=r"C:\Users\Administrator\Desktop\English-Jar\lei\output\Las Vegas_20core\reviews.pickle"
reviews=pickle.load(open(path_reviews,"rb"))
reviews=Replace_ID_with_Index.replace_id_with_index(reviews=reviews,userid_to_index=userid_to_index,itemid_to_index=itemid_to_index)
unique_features=list()
for review in reviews:
    if "sentence" not in review.keys():continue
    feature=review["sentence"][0][0]
    if feature not in unique_features:
        unique_features.append(feature)
p=len(unique_features)
get_matrices_object=Get_Matrices.get_matrices(reviews=reviews,m=m,n=n,p=p,N=R_max)
get_matrices_object.user_feature_attention_matrix()
get_matrices_object.item_feature_quality_matrix()
X=get_matrices_object.X
Y=get_matrices_object.Y
"""
Step 4：Compute the opinion scores for training, validation, and testing. 
"""
opin_scores_train=[]
opin_scores_validation=[]
opin_scores_test=[]
for interaction in df_train.itertuples():
    user=getattr(interaction,"user_index")
    item=getattr(interaction,"item_index")
    feature_based_score=np.dot(X[user],Y[item])
    opin_scores_train.append(feature_based_score)
for interaction in df_validation.itertuples():
    user=getattr(interaction,"user_index")
    item=getattr(interaction,"item_index")
    feature_based_score=np.dot(X[user],Y[item])
    opin_scores_validation.append(feature_based_score)
for interaction in df_test.itertuples():
    user=getattr(interaction,"user_index")
    item=getattr(interaction,"item_index")
    feature_based_score=np.dot(X[user],Y[item])
    opin_scores_test.append(feature_based_score)

"""
Step 4: Optimal hyperparameter search via grid search cross validation
"""
"""
For each hp_combination stored as a two-dimensional numpy array, 
the first entry is a choice for the regularization coefficient (lambda),
the second entry is a choice for the number of latent factors (k),
the third entry is a choice for the rating-opinion relative contribution weight (c)
"""
path_USMF_algorithm_training=r"C:\Users\Administrator\Desktop\上科大\代码\USMF algorithm training"
sys.path.append(path_USMF_algorithm_training)
import Hyperparameters as hp
import HyperparameterTuning_TrainValidationSplit_USMF as ht_trainvalidation
param_grid={"regularization coefficient":[0.1,0.01,0.001,1e-4,1e-5],
            "number of latent factors":[20,40,60,80,100],
            "rating-opinion relative contribution weight":[0,0.2,0.5,0.8,1]}
hp_combinations=hp.hyperparameter_combinations(param_grid)
hp_train_validation_split=ht_trainvalidation.hyperparametertuning_train_validation_split_USMF(df_train=df_train,df_validation=df_validation,expo_prob_train=expo_prob_train,expo_prob_validation=expo_prob_validation,opin_scores_train=opin_scores_train,hp_combinations=hp_combinations)
#hp_train_validation_split.train_validation_split_unbiasedRMSE() # hyperparameter tuning via unbiased RMSE
hp_train_validation_split.train_validation_split_unbiasedNDCG() # hyperparameter tuning via unbiased NDCG
#k_validation=3 # test the model's performance on the validation set through top-k_validation recommendation
#hp_train_validation_split.train_validation_split_unbiasedrecall(k=k_validation) # hyperparameter tuning via unbiased recall

# find the optimal hyperparameter combination evaluated on unbiased RMSE
#optimal_hp_combination_unbiasedRMSE=hp_train_validation_split.hp_combinations[np.argmin(hp_train_validation_split.hp_combination_performance_unbiasedRMSE)]

# find the optimal hyperparameter combination evaluated on unbiased NDCG
optimal_hp_combination_unbiasedNDCG=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedNDCG)]

# find the optimal hyperparameter combination evaluated on unbiased recall
optimal_hp_combination_unbiasedrecall=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedrecall)]

"""
Step 5: USMF training, testing and performance evaluation for top-K recommendation
"""
df_train=pd.concat([df_train,df_validation],axis=0)
expo_prob_train=expo_prob_train+expo_prob_validation
expo_prob_test=expo_prob_test
opin_scores_train=opin_scores_train+opin_scores_validation
from USMF_2 import USMF
usmf=USMF(df_train,df_test,expo_prob_train,expo_prob_test,opin_scores_train)


# compute the model's performance on the test set evluated via unbiased NDCG
#unbiased_RMSE=usmf.performance_unbiasedRMSE(optimal_hp_combination_unbiasedRMSE)
#print("The model's performance evaluated via unbiased RMSE is {}".format(unbiased_RMSE))

# compute the model's performance on the test set evluated via unbiased NDCG
unbiased_NDCG=usmf.performance_unbiasedNDCG(optimal_hp_combination_unbiasedNDCG)
print("The model's performance evaluated via unbiased NDCG is {}".format(unbiased_NDCG))

# compute the model's performance on the test set evaluated via unbiased recall
#k_test=3 # test the model's performance on the test set through top-k_test recommendation
#unbiased_recall=usmf.performance_unbiasedrecall(optimal_hp_combination_unbiasedrecall,k=k_test)
#print("The model's performance evaluated via unbiased recall is {}".format(unbiased_recall))


