#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:23:44 2020

@author: nuoyuan
"""
import nunpy as np

class TrainingAlgorithm(object):
    def __init__(self,hp_combination,universal_average):
        """
        Parameters
        ----------
        user_feature: A Python dictionary each key-value pair of which corresponds to a user (the key is the index for that user) and their low-dimensional feature (the value is the feature for that user)
        user_bias: A Python dictionary each key-value pair of which corresponds to a user (the key is the index for that user) and their rating bias (the value is the rating bias for that user)
        item_feature: A Python dictionary each key-value pair of which corresponds to an item (the key is the index for that item) and their low-dimensional feature (the value is the feature for that item)
        item_bias: A Python dictionary each key-value pair of which corresponds to an item (the key is the index for that item) and their rating bias (the value is the rating bias for that item)
        hp_combination: A numpy array representing some hyperparameter combination
            
        Returns
        -------
        None.
        """
        self.user_feature={}
        self.user_bias={}
        self.item_feature={}
        self.item_bias={}
        self.hp_combination=hp_combination
        self.universal_average=universal_average #universal_average refers to the arithmetic average of all the relevance predictions contained in the training set
    def sgd(self,df_train,step=100):
        """
        This function trains the USMF recommendation learning algorithm by minimizing the loss function using the stochastic gradient descent approach. Taking in df_train, hp_combination, step as its input parameters,
        this function outputs the user features, the user rating biases, the item features, and the item rating biases
        Parameters
        ----------
        df_train : The traing set stored as a dataframe
        step : The maximum number of iterations set to guarantee convergence. The default is 100
           
        Returns
        -------
        none
        """
        reg_coe=self.hp_combination[0] # assume that the index of the regularization coefficient to prevent overfitting in hp_combination is 0
        num_factors=self.hp_combination[1] # assume that the index of the number of latent factors in hp_combination is 1
        t0=5 # t0 is the numerator of the fraction to generate iteration-specific learning rate inspired by simulated annealing
        t1=50 # t1 is part of the denominator of the fraction to generate iteration-specific learning rate inspired by simulated annealing
        def learning_rate(t):
            return t0/(t1+t)
        for iteration in range(step):
            learning_rate=learning_rate(iteration)
            for row in df_train.itertuples():
                u,i,r,e=getattr(row,"user_index"),getattr(row,"item_index"),getattr(row,"rating"),getattr(row,"exposure_probability")
                # initialize the user feature
                if u not in self.user_feature:
                    self.user_feature[u]=np.random.rand(num_factors)
                # initialize the user rating bias
                if u not in self.user_bias:
                    self.user_bias[u]=np.random.rand(1)
                # initialize the item feature
                if i not in self.item_feature:
                    self.item_feature[i]=np.random.rand(num_factors)
                # initialize the item rating bias
                if i not in self.item_bias:
                    self.item_bias[i]=np.random.rand(1)
                error=r-(self.universal_average+np.dot(self.user_feature[u],self.item_feature[i]))
                # update the user feature
                self.user_feature[u]+=learning_rate*((error/e)*self.item_feature[i]-reg_coe*self.user_feature[u])
                # update the user rating bias
                self.user_bias[u]+=learning_rate*(error/e)
                # update the item feature
                self.item_feature[i]+=learning_rate*((error/e)*self.user_feature[u]-reg_coe*self.item_feature[i])
                # update the item rating bias
                self.item_bias[i]+=learning_rate*(error/e)
                    
                    
        