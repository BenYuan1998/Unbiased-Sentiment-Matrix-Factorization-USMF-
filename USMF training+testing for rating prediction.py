#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:39:44 2020

@author: nuoyuan
"""

from renew_trainingset import renew_trainingset
from TrainingAlgorithm import TrainingAlgorithm
import numpy as np
import pandas as pd


df_train_ratingpred=pd.DataFrame({}) # the training set for rating prediction
df_test_ratingpred=pd.DataFrame({}) # the test set for rating prediction
optimal_hpcombination=np.array([]) # the optimal hyperparameter combination determined by 5-fold grid search cross validation
c=optimal_hpcombination[3]
expo_prob=[] # This list needs to be computed
opin_scores=[] # This list needs to be computed
df_train_ratingpred=renew_trainingset(df_train_ratingpred,expo_prob,opin_scores,c)
universal_average=df_train_ratingpred["rating"].values.sum()/df_train_ratingpred.shape[0]

# Train USMF on the training set
trainingalgorithm=TrainingAlgorithm(optimal_hpcombination,universal_average)
step=100
trainingalgorithm.sgd(df_train_ratingpred,step)

# Test the trained USMF on the test set for rating prediction
unbiased_RMSE_sum_over_items=0
unique_item_index=np.unique(df_test_ratingpred["item_index"].values).tolist()
num_items=len(unique_item_index)
for itemj in unique_item_index:
    itemj_SE=0
    itemj_indices=df_test_ratingpred[df_test_ratingpred["item_index"]==itemj].index.tolist()
    num_interactions=len(itemj_indices)
    df_test_ratingpred_itemj=df_test_ratingpred[itemj_indices]
    for user_itemj_interaction in df_test_ratingpred_itemj.itertuples():
        user_index=getattr(user_itemj_interaction,"user_index")
        item_index=itemj
        rating_actual=getattr(user_itemj_interaction,"rating")
        rating_predicted=universal_average+trainingalgorithm.user_bias[user_index]+trainingalgorithm.item_bias[item_index]+np.dot(trainingalgorithm.user_feature[user_index],trainingalgorithm.item_feature[item_index])
        error_squared=(rating_actual-rating_predicted)**2
        itemj_SE+=error_squared
    itemj_RMSE=np.sqrt((itemj_SE/num_interactions))
    unbiased_RMSE_sum_over_items+=itemj_RMSE
unbiased_RMSE_average=unbiased_RMSE_sum_over_items/num_items
        
    