# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:29:14 2020

@author: Administrator
"""

import sys
import os
import numpy as np
import pandas as pd


"""
This script pre-processes the subject dataset to split it into a training set and a test set, finds the 
optimal hyperparameter combination through grid search cross validation, and trains the MF-Opin model on
the training set and tests the trained model on the test set to assess the model's ranking performance via
unbiased DCG/recall.
"""

"""
Step 1: split the subject dataset into a training and a test set
"""
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
test_size=5 
df_train_validation=Train_Test_Split.training_set(df_ncore,m=test_size)
training_validation_indices=df_train_validation.index.tolist()
df_test=Train_Test_Split.test_set(df_ncore,training_validation_indices)
validation_size=5   
df_train=Train_Test_Split.training_set(df_train_validation,m=validation_size)
training_indices=df_train.index.tolist()
df_validation=Train_Test_Split.test_set(df_train_validation,training_indices)

"""
Step 2: propensity scores' estimation for the training set and test set
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
Step 3: Opinion scores' computation
"""
path_Opinion_Matrix=r"C:\Users\Administrator\Desktop\上科大\代码\Opinion Matrix"
sys.path.append(path_Opinion_Matrix)
import Feature_Level_Sentiment_Analysis, Replace_ID_with_Index
import pickle
path_reviews=r"C:\Users\Administrator\Desktop\English-Jar\lei\output\Las Vegas_20core\reviews.pickle"
reviews=pickle.load(open(path_reviews,"rb"))
reviews=Replace_ID_with_Index.replace_id_with_index(reviews=reviews,userid_to_index=userid_to_index,itemid_to_index=itemid_to_index)
# Find the user-item index pairs in the training set, the validation set, and the test set respectively.
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


# Compute the opinion scores for all the reviews contained in the training set based on feature-level sentiment analysis.
flsa_object=Feature_Level_Sentiment_Analysis.feature_level_sentiment_analysis(reviews=reviews_train,df=df_train)
features_unique=flsa_object.unique_feature_extraction()
X=flsa_object.user_feature_attention_matrix()
Y=flsa_object.item_feature_quality_matrix()
opin_scores_train=flsa_object.feature_level_opinion_score()
# compute the opinin scores for all the reviews contained in the validation set based on feature-level sentiment analysis.
flsa_object=Feature_Level_Sentiment_Analysis.feature_level_sentiment_analysis(reviews=reviews_validation,df=df_validation)
features_unique=flsa_object.unique_feature_extraction()
X=flsa_object.user_feature_attention_matrix()
Y=flsa_object.item_feature_quality_matrix()
opin_scores_validation=flsa_object.feature_level_opinion_score()
# Compute the opinion scores for all the reviews contained in the test set based on feature-level sentiment analysis.
flsa_object=Feature_Level_Sentiment_Analysis.feature_level_sentiment_analysis(reviews=reviews_test,df=df_test)
features_unique=flsa_object.unique_feature_extraction()
X=flsa_object.user_feature_attention_matrix()
Y=flsa_object.item_feature_quality_matrix()
opin_scores_test=flsa_object.feature_level_opinion_score()
"""
Step 4: Optimal hyperparameter search via grid search cross validation
"""
"""
For each hp_combination stored as a two-dimensional numpy array, 
the first entry is a choice for the regularization coefficient (lambda),
the second entry is a choice for the number of latent factors (k),
the third entry is a choice for the rating-opinion relative contribution weight (c)
"""
path_MF_Opin_algorithm_training=r"C:\Users\Administrator\Desktop\上科大\代码\Baselines\MF-Opin"
sys.path.append(path_MF_Opin_algorithm_training)
import Hyperparameters as hp
import HyperparameterTuning_TrainValidationSplit_MF_Opin as ht_trainvalidation
param_grid={"regularization coefficient":[0.1,0.01,0.001,1e-4,1e-5],
            "number of latent factors":[20,40,60,80,100],
            "rating-opinion relative contribution weight":[0,0.2,0.5,0.8,1]}
hp_combinations=hp.hyperparameter_combinations(param_grid)
hp_train_validation_split=ht_trainvalidation.hyperparametertuning_train_validation_split_MF_Opin(df_train=df_train,df_validation=df_validation,expo_prob_train=expo_prob_train,expo_prob_validation=expo_prob_validation,opin_scores_train=opin_scores_train,opin_scores_validation=opin_scores_validation,hp_combinations=hp_combinations)
#hp_train_validation_split.train_validation_split_unbiasedNDCG() # hyperparametr tuning via unbiased NDCG
k_validation=3 # test the model's performance on the validation set through top-k_validation recommendation
hp_train_validation_split.train_validation_split_unbiasedrecall(k=k_validation) # hyperparameter tuning via unbiased recall

# find the optimal hyperparameter combination evaluated on unbiased NDCG
#optimal_hp_combination_unbiasedNDCG=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedNDCG)]

# find the optimal hyperparameter combination evaluated on unbiased recall
optimal_hp_combination_unbiasedrecall=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedrecall)]

"""
Step 5: MF_Opin training, testing and performance evaluation for top-K recommendation
"""
df_train=pd.concat([df_train,df_validation],axis=0)
expo_prob_train=expo_prob_train+expo_prob_validation
expo_prob_test=expo_prob_test
opin_scores_train=opin_scores_train+opin_scores_validation
opin_scores_test=opin_scores_test
from MF_Opin import MF_Opin
mf_opin=MF_Opin(df_train,df_test,expo_prob_train,expo_prob_test,opin_scores_train,opin_scores_test)

# compute the model's performance on the test set evluated via unbiased NDCG

#unbiased_NDCG=mf_opin.performance_unbiasedNDCG(optimal_hp_combination_unbiasedNDCG)
#print("The model's performance evaluated via unbiased nDCG is {}".format(unbiased_NDCG))

# compute the model's performance on the test set evaluated via unbiased recall
k_test=3 # test the model's performance on the test set through top-k_test recommendation
unbiased_recall=mf_opin.performance_unbiasedrecall(optimal_hp_combination_unbiasedrecall,k=k_test)
print("The model's performance evaluated via unbiased recall is {}".format(unbiased_recall))