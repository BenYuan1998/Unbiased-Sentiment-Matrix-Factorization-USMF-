#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:15:42 2020

@author: nuoyuan
"""
from hyperparameters import hyperparameter_combinations
from renew_trainingset import renew_trainingset
from TrainingAlgorithm import TrainingAlgorithm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

param_grid={} # this empty dictionary should be replaced by an actual param_grid
hp_combinations=hyperparameter_combinations(param_grid)
hpcombination_performances=list() # A Python built-in list storing the model's average performance with each hyperparameter combination in K-fold cross validation
for hp_combination in hp_combinations:
    # pick a hyperparameter combination (i.e., a row of hp_combinations) from the set of all possible combinations 
    c=hp_combination[3] # Assume that the index of the rating-opinion weight in hp_combination is 3
    df_train=pd.DataFrame.from_dict({}) # df_train=training_set(df_new,n) (in the Data pre-processing folder)
    expo_prob=[] # This list needs to be computed
    opin_scores=[] # This list needs to be computed
    df_train_new=renew_trainingset(df_train,expo_prob,opin_scores,c)
    # divide the training set into K approximately equal-size folds
    K=5
    kf=KFold(n_splits=K,shuffle=True)
    hpcombination_performance_sum_over_folds=0 # the performance of a hyperparameter combination is computed as the arithmetic average of the fold-level model errors.
    for train_indices,test_indices in kf.split(df_train_new):
        # df_train_gscv: the traning set for hyperparameter tuning using grid search cross validation
        # df_test_gscv: the test set for hyperparameter tuning using grid search cross validation
        df_train_gscv=df_train_new.loc[train_indices]
        df_test_gscv=df_train_new.loc[test_indices]
        universal_average=df_train_gscv["rating"].values.sum()/df_train_gscv.shape[0]
        trainingalgorithm=TrainingAlgorithm(hp_combination,universal_average)
        # train the USMF model on df_train_gscv until model convergence/the number of iterations reaches the number indicated by step
        step=100
        trainingalgorithm.sgd(df_train_gscv,step)
        # assess the model's performance on RMSE by computing the arithmetic average of the item-specific RMSEs associated with all the items in the test set for unbiased evaluation
        unbiased_RMSE_sum_over_items=0
        unique_item_indices_test_gscv=np.unique(df_test_gscv["item_index"].values).tolist()
        num_items=len(unique_item_indices_test_gscv)
        for itemj in unique_item_indices_test_gscv:
            unbiased_SE_itemj=0
            itemj_indices=df_test_gscv[df_test_gscv["item_index"]==itemj].index.tolist()
            num_interactions=len(itemj_indices)
            for user_itemj_interaction in df_test_gscv.loc[itemj_indices].itertuples():
                user_index=getattr(user_itemj_interaction,"user_index")
                item_index=itemj
                rating_actual=getattr(user_itemj_interaction,"rating")
                rating_predicted=universal_average+trainingalgorithm.user_bias[user_index]+trainingalgorithm.item_bias[item_index]+np.dot(trainingalgorithm.user_feature[user_index],trainingalgorithm.item_feature[item_index])
                error_squared=(rating_actual-rating_predicted)**2
                unbiased_SE_itemj+=error_squared
            unbiased_RMSE_itemj=np.sqrt((unbiased_SE_itemj/num_interactions))
            unbiased_RMSE_sum_over_items+=unbiased_RMSE_itemj
        unbiased_RMSE_average=unbiased_RMSE_sum_over_items/num_items
        hpcombination_performance_sum_over_folds+=unbiased_RMSE_average
    hpcombination_performances.append(hpcombination_performance_sum_over_folds/K)
optimal_hpcombination_performance_unbiasedRMSE=hpcombination_performances[np.argmin(hpcombination_performances)]
            
                    
                    
                    
                    
                
