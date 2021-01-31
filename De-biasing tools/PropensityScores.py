#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:25:11 2020

@author: nuoyuan
"""

import pandas as pd
import numpy as np

class PropensityScores(object):
    """
    This class is a collection of different approaches towards estimating propensity scores
    """
    def __init__(self,df_train):
        self.df_train=df_train
        self.expo_prob=list()
    def user_independent_PS(self,power):
        """
        Parameters
        ----------
        power : The exponent of the power-law distribution model used for user-independent propensity
                score estimation. A larger power leads to lower propensity score estimates for the 
                long-tail items and higher propensity score estimates for the popular ones.
        Returns
        -------
        None.
        """
        observed_interactions_dict=dict() # a Python built-in dictionary storing each item index as a key and the item's number of observed interactions as its corresponding value
        unique_item_indices=np.unique(self.df_train["item_index"].values).tolist()
        for itemj in unique_item_indices: 
            popularity_observed=self.df_train[self.df_train["item_index"]==itemj].shape[0]
            observed_interactions_dict[itemj]=popularity_observed
        most_popular_item_interactions=max(observed_interactions_dict.values())
        for key in observed_interactions_dict.keys():
            observed_interactions_dict[key]=(observed_interactions_dict[key]/most_popular_item_interactions)**power
        expo_prob_dict=observed_interactions_dict  # the propensity score for each item is estimated as its relative observed popularity raised to the power of the number specified by the parameter
        for item in self.df_train["item_index"].values:
            for unique_item in expo_prob_dict.keys():
                if item==unique_item:
                    self.expo_prob.append(expo_prob_dict[unique_item])
            