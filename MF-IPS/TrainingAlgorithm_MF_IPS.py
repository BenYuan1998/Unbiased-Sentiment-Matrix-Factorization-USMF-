# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:21:36 2020

@author: Administrator
"""
import numpy as np
class trainingalgorithm_MF_IPS(object):
    
    def __init__(self,hp_combination):
        """

        Parameters
        ----------
        hp_combination: A numpy array representing some hyperparameter combination

        Returns
        -------
        None.
        """
        self.hp_combination=hp_combination
        self.user_feature=dict()
        self.item_feature=dict()
    def sgd(self, df_train, step=100):
        reg_coe=self.hp_combination[0]
        num_factors=int(self.hp_combination[1])
        t0=5 # t0 is the numerator of the fraction to generate iteraction-specific learning rate inspired by simulated annealing 
        t1=5000 # t1 is the denominator of the fraction to generate iteration-specific learning rate inspired by simulated annealing
        def learning_rate(t):
            return t0/(t+t1)
        for iteration in range(step):
            learningrate=learning_rate(iteration)
            for row in df_train.itertuples():
                u,i,r,e=getattr(row,"user_index"),getattr(row,"item_index"),getattr(row,"rating"),getattr(row,"exposure_probability")
                # initialize the user feature
                if u not in self.user_feature:
                    self.user_feature[u]=np.random.rand(num_factors)
                # initialize the item feature
                if i not in self.item_feature:
                    self.item_feature[i]=np.random.rand(num_factors)
                error=r-np.dot(self.user_feature[u],self.item_feature[i])
                # update the user feature
                self.user_feature[u]+=learningrate*((error/e)*self.item_feature[i]-reg_coe*self.user_feature[u])
                # update the item feature
                self.item_feature[i]+=learningrate*((error/e)*self.user_feature[u]-reg_coe*self.item_feature[i])
    
         