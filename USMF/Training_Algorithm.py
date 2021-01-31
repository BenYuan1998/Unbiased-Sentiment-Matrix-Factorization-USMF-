# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:32 2021

@author: Administrator
"""
import sys
import numpy as np
path_general_functions=r"C:\Users\Administrator\Desktop\上科大\代码\General functions"
sys.path.append(path_general_functions)

class trainingalgorithm_USMF(object):
    def __init__(self,hp_combination,average_rating):
        """
        Parameters
        ----------
        hp_combination: A numpy array representing some hyperparameter combination
        average_rating: The average of all the ratings in the training set
        Returns
        -------
        None.
        """
        self.hp_combination=hp_combination
        self.average_rating=average_rating
        self.user_feature={}
        self.item_feature={}
        self.user_bias={}
        self.item_bias={}
    def sgd(self,df_train,X,Y,step=100):
        """
        This function trains the USMF recommendation learning algorithm by minimizing the loss function using the stochastic gradient descent approach. Taking in df_train, hp_combination, step as its input parameters,
        this function outputs the user features, the user rating biases, the item features, and the item rating biases
        Parameters
        ----------
        df_train : The traing set stored as a dataframe
        step : The maximum number of iterations set to guarantee convergence. The default is 100.
           
        Returns
        -------
        none
        """
        reg_coe_factors=self.hp_combination[0] 
        reg_coe_biases=self.hp_combination[1]
        c=self.hp_combination[2]
        num_factors=X.shape[1]
        t0=5 # t0 is the numerator of the fraction to generate iteration-specific learning rate inspired by simulated annealing
        t1=5000 # t1 is part of the denominator of the fraction to generate iteration-specific learning rate inspired by simulated annealing
        iterations=list()
        costs=list()
        def learning_rate(t):
            return t0/(t1+t)
        for iteration in range(step):
            E=0
            learningrate=learning_rate(iteration)
            for row in df_train.itertuples():
                u,i,r,e=getattr(row,"user_index"),getattr(row,"item_index"),getattr(row,"rating"),getattr(row,"exposure_probability")
                u_preference=X[u] # the p-dimensional preference vector for user u
                i_performance=Y[i] # the p-dimensional performance vector for item i
                # initialize the user feature
                if u not in self.user_feature:
                    self.user_feature[u]=np.random.rand(num_factors)
                # initialize the item feature
                if i not in self.item_feature:
                    self.item_feature[i]=np.random.rand(num_factors)
                # initialize the user bias
                if u not in self.user_bias:
                    self.user_bias[u]=np.random.rand(1)[0]
                # initialize the item bias
                if i not in self.item_bias:
                    self.item_bias[i]=np.random.rand(1)[0]
                error_per_sample=r-(self.average_rating+self.user_bias[u]+self.item_bias[i]+np.dot((c*self.user_feature[u]+(1-c)*u_preference),(c*self.item_feature[i]+(1-c)*i_performance)))
                E+=error_per_sample**2/e
                # update the user feature
                self.user_feature[u]+=learningrate*((error_per_sample/e)*(np.square(c)*self.item_feature[i]+c*(1-c)*i_performance)-reg_coe_factors*self.user_feature[u])
                # update the item feature
                self.item_feature[i]+=learningrate*((error_per_sample/e)*(np.square(c)*self.user_feature[u]+c*(1-c)*u_preference)-reg_coe_factors*self.item_feature[i])
                # update the user bias
                self.user_bias[u]+=learningrate*((error_per_sample/e)-reg_coe_biases*self.user_bias[u])
                # update the item bias
                self.item_bias[i]+=learningrate*((error_per_sample/e)-reg_coe_biases*self.item_bias[i])
            L2_norm_user_preferences=0
            L2_norm_item_attributes=0
            L2_norm_user_biases=0
            L2_norm_item_biases=0
            for user in self.user_feature.keys():
                L2_norm_user_preferences+=np.sum(np.square(self.user_feature[user]))
            for item in self.item_feature.keys():
                L2_norm_item_attributes+=np.sum(np.square(self.item_feature[user]))
            for user in self.user_bias.keys():
                L2_norm_user_biases+=np.square(self.user_bias[user])
            for item in self.item_bias.keys():
                L2_norm_item_biases+=np.square(self.item_bias[item])
            E_regularization=reg_coe_factors*(L2_norm_user_preferences+L2_norm_item_attributes)+reg_coe_biases*(L2_norm_user_biases+L2_norm_item_biases)
            E=E+E_regularization
            iterations.append(iteration+1)
            costs.append(E)
        return[iterations,costs]