#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:35:20 2020

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
         # assess the model's performance on NDCG by incorporating exposure probability estimates for unbiased evaluation
        unbiased_NDCG_sum_over_users=0
        unique_user_indices_test_gscv=np.unique(df_test_gscv["user_index"].values).tolist()
        num_users=len(unique_user_indices_test_gscv)
        for useri in unique_user_indices_test_gscv:
            useri_indices=df_test_gscv[df_test_gscv["user_index"]==useri].index.tolist()
            df_test_gscv_useri=df_test_gscv.loc[useri_indices] # the subset of the validation fold (a dataframe) exclusively composed of all the user i-item interactions
            useri_list_rating_predicted=list()
            useri_list_rating_actual=list()
            useri_list_expo_prob=list()
            for useri_item_interaction in df_test_gscv_useri.itertuples:
                user_index=useri
                item_index=getattr(useri_item_interaction,"item_index")
                expo_prob=getattr(useri_item_interaction,"exposure_probability")
                rating_actual=getattr(useri_item_interaction,"rating")
                rating_predicted=universal_average+trainingalgorithm.user_bias[user_index]+trainingalgorithm.item_bias[item_index]+np.dot(trainingalgorithm.user_feature[user_index],trainingalgorithm.item_feature[item_index])
                # compute the actual unbiased DCG for the ranked item list recommended by USMF to useri
                useri_list_expo_prob.append(expo_prob)
                useri_list_rating_actual.append(rating_actual)
                useri_list_rating_predicted.append(rating_predicted)
            def ranking_based_on_rating(list_rating):
                """
                Parameters
                ----------
                list_rating : A Python built-in list of ratings

                Returns
                -------
                list_ranking: A Python built-in list of size equal to that of list_rating, 
                containing rank scores computed in a descending fashion (i.e., 
                the largest value in list_rating is assigned a rank score of 1) using the "competition ranking" method 
                (i.e.,  The minimum of the ranks that would have been assigned to all the tied values is assigned to each value.)
                """
                from scipy.stats import rankdata
                list_ranking_reversed=rankdata(list_rating,method="max")
                list_ranking=list()
                num_elements=len(list_rating)
                for reversed_rank in list_ranking_reversed:
                    list_ranking.append(num_elements-reversed_rank+1)
                return list_ranking
            def get_unbiasedDCG(ranking_predicted,rating_actual,expo_prob):
                """
                Parameters
                ----------
                ranking_predicted : A Python built-in list of rank scores for the predicted ratings on a select group of items generated by a recommendation system
                rating_actual : A Python built-in list of the actual ratings on the same group of items as those associated with ranking_predicted
                expo_prob : A Python built-in list of the exposure probability estimates associated with the same group of items as those associated with ranking_predicted
                Returns
                -------
                unbiasedDCG: The unbiased DCG for the recommended list
                """
                unbiasedDCG=0
                for index,ranking in enumerate(ranking_predicted):
                    unbiasedDCG+=((2**rating_actual[index]-1)/np.log2(ranking+1))/expo_prob[index]
                return unbiasedDCG
            # compute the actual unbiased DCG for the ranked item list recommended by USMF to useri (i.e., actual_unbiasedDCG)
            useri_list_ranking_predicted=ranking_based_on_rating(useri_list_rating_predicted)
            actual_unbiasedDCG=get_unbiasedDCG(useri_list_ranking_predicted,useri_list_rating_actual,useri_list_expo_prob)
            # compute the unbiased DCG for the ideal ranked item list for useri (i.e., ideal_unbiasedDCG)
            useri_list_ranking_actual=ranking_based_on_rating(useri_list_rating_actual)
            ideal_unbiasedDCG=get_unbiasedDCG(useri_list_ranking_actual,useri_list_rating_actual,useri_list_expo_prob)
            # compute the negative unbiased NDCG for the ranked item list recommended by USMF to useri 
            unbiasedNDCG=actual_unbiasedDCG/ideal_unbiasedDCG
            unbiased_NDCG_sum_over_users+=unbiasedNDCG
        unbiased_NDCG_average=unbiased_NDCG_sum_over_users/num_users
        hpcombination_performance_sum_over_folds+=unbiased_NDCG_average
    hpcombination_performances.append(hpcombination_performance_sum_over_folds/K)
optimal_hpcombination_unbiasedNDCG=hpcombination_performances[np.argmax(hpcombination_performances)]
    
        
