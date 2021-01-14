# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:33:28 2020

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

"""
Step 1: split the subject dataset into a training and a test set
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
test_size=5 # m stands for the number of most recently interacted items for each user to be included in the test set. 
df_train=Train_Test_Split.training_set(df_ncore,m=test_size)
training_indices=df_train.index.tolist()
df_test=Train_Test_Split.test_set(df_ncore,training_indices)

"""
Step 2: propensity scores' estimation for the training set and test set
"""
path_debiasing_tools=r"C:\Users\Administrator\Desktop\上科大\代码\De-biasing tools"
sys.path.append(path_debiasing_tools)
import PropensityScores as PS
propensityscores_train=PS.PropensityScores(df_train)
propensityscores_test=PS.PropensityScores(df_test)
power=0.5 # the power of the power-law distribution for the user-independent approach to propensity scores' estimation is set as 0.5
propensityscores_train.user_independent_PS(power)
propensityscores_test.user_independent_PS(power)
expo_prob_train=propensityscores_train.expo_prob
expo_prob_test=propensityscores_test.expo_prob

"""
Step 3: Optimal hyperparameter search via grid search cross validation
"""
"""
For each hp_combination stored as a two-dimensional numpy array, 
the first entry is a choice for the regularization coefficient (lambda),
the second entry is a choice for the number of latent factors (k),
"""
path_MF_IPS=r"C:\Users\Administrator\Desktop\上科大\代码\Baselines\MF-IPS"
sys.path.append(path_MF_IPS)
import Hyperparameters as hp
import HyperparameterTuning_TrainValidationSplit_MF_IPS as ht_trainvalidation
param_grid={"regularization coefficient":[0.1,0.01,0.001,1e-4,1e-5],
            "number of latent factors":[20,40,60,80,100]
            }
hp_combinations=hp.hyperparameter_combinations(param_grid)
validation_size=5 # the number of most recently occurred user-item inteactions for each user to be contained in the validation set.
hp_train_validation_split=ht_trainvalidation.hyperparametertuning_train_validation_split_MF_IPS(df_train,expo_prob_train,hp_combinations,m=validation_size)

hp_train_validation_split.train_validation_split_unbiasedNDCG() # hyperparameter tuning via unbiased NDCG
k_validation=3 # test the model's performance on the validation set via top-k_validation recommendation
hp_train_validation_split.train_validation_split_unbiasedrecall(k=k_validation) # hyperparameter tuning via unbiased recall

# find the optimal hyperparameter combination evaluated on unbiased NDCG
#optimal_hp_combination_unbiasedNDCG=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedNDCG)]

# find the optimal hyperparameter combination evaluated on unbiased recall
optimal_hp_combination_unbiasedrecall=hp_train_validation_split.hp_combinations[np.argmax(hp_train_validation_split.hp_combination_performance_unbiasedrecall)]

"""
Step 4: MF-IPS training, testing and performance evaluation for top-K recommendation
"""

from MF_IPS import MF_IPS
mf_ips=MF_IPS(df_train,df_test,expo_prob_train,expo_prob_test)

# compute the model's performance on the test set evluated via unbiased NDCG
#unbiased_NDCG=mf_ips.performance_unbiasedNDCG(optimal_hp_combination_unbiasedNDCG)
#print("The model's performance evaluated via unbiased NDCG is {}".format(unbiased_NDCG))

# compute the model's performance on the test set evaluated via unbiased recall
k_test=3 # test the model's performance on the test set via top-k_test recommendation
unbiased_recall=mf_ips.performance_unbiasedrecall(optimal_hp_combination_unbiasedrecall,k=k_test)
print("The model's performance evaluated via unbiased recall is {}".format(unbiased_recall))