#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:23:44 2020

@author: nuoyuan
"""
import numpy as np

class trainingalgorithm_USMF(object):
    def __init__(self,hp_combination):
        """
        Parameters
        ----------
        user_feature: A Python dictionary each key-value pair of which corresponds to a user (the key is the index for that user) and their low-dimensional representation (the value is the feature for that user)
        user_bias: A Python dictionary each key-value pair of which corresponds to a user (the key is the index for that user) and their rating bias (the value is the rating bias for that user)
        item_feature: A Python dictionary each key-value pair of which corresponds to an item (the key is the index for that item) and their low-dimensional representation (the value is the feature for that item)
        item_bias: A Python dictionary each key-value pair of which corresponds to an item (the key is the index for that item) and their rating bias (the value is the rating bias for that item)
        hp_combination: A numpy array representing some hyperparameter combination
            
        Returns
        -------
        None.
        """
        self.hp_combination=hp_combination
        self.user_feature={}
        self.item_feature={}
    def checking_convergence(iterations,costs):
        import matplotlib.pyplot as plt
        plt.plot(iterations,costs)
        plt.xlabel("ith iteration")
        plt.ylabel("value of the loss function")
        plt.show()
    def sgd(self,df_train,step=500):
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
        num_factors=int(self.hp_combination[1]) # assume that the index of the number of latent factors in hp_combination is 1
        t0=5 # t0 is the numerator of the fraction to generate iteration-specific learning rate inspired by simulated annealing
        t1=50000 # t1 is part of the denominator of the fraction to generate iteration-specific learning rate inspired by simulated annealing
        iterations=list()
        costs=list()
        def learning_rate(t):
            return t0/(t1+t)
        for iteration in range(step):
            E=0
            learningrate=learning_rate(iteration)
            for row in df_train.itertuples():
                u,i,r,e=getattr(row,"user_index"),getattr(row,"item_index"),getattr(row,"relevance"),getattr(row,"exposure_probability")
                # initialize the user feature
                if u not in self.user_feature:
                    self.user_feature[u]=np.random.rand(num_factors)
                # initialize the item feature
                if i not in self.item_feature:
                    self.item_feature[i]=np.random.rand(num_factors)
                error_per_sample=r-np.dot(self.user_feature[u],self.item_feature[i])
                E+=error_per_sample**2/e
                # update the user feature
                self.user_feature[u]+=learningrate*((error_per_sample/e)*self.item_feature[i]-reg_coe*self.user_feature[u])
                # update the item feature
                self.item_feature[i]+=learningrate*((error_per_sample/e)*self.user_feature[u]-reg_coe*self.item_feature[i])
            L2_norm_user_preferences=0
            L2_norm_item_attributes=0
            for user in self.user_feature.keys():
                L2_norm_user_preferences+=np.sum(np.square(self.user_feature[user]))
            for item in self.item_feature.keys():
                L2_norm_item_attributes+=np.sum(np.square(self.item_feature[user]))
            E_regularization=reg_coe*(L2_norm_user_preferences+L2_norm_item_attributes)
            E=E+E_regularization
            iterations.append(iteration+1)
            costs.append(E)
        return[iterations,costs]

               
                    
                    
        