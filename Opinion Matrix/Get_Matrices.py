# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:39:29 2021

@author: Administrator
"""

import numpy as np
class get_matrices(object):
    def __init__(self,reviews,m,n,p,N):
        """
        Parameters
        ----------
        reviews : A list of dictionaries with the following key-value structure:
                'user': user_index,
                'item': item_index,
                'rating': numerical rating,
                'text': the textual review,
                'sentence': the feature-opinion-sentence-polarity quadruple.
        m: the number of users.
        n: the number of items.
        p: the number of feature phrases.
        N：the greatest possible number on the numerical rating scale.  
        Returns
        -------
        None.
        """
        self.reviews=reviews
        self.m=m
        self.n=n
        self.p=p
        self.N=N
        self.A=np.zeros((m,n))
        self.X=np.zeros((m,p))
        self.Y=np.zeros((n,p))
    def feature_index_pair(self):
        features=list()
        for review in self.reviews:
            if "sentence" not in review.keys():continue
            feature=review["sentence"][0][0]
            features.append(feature)
        unique_features=np.unique(features).tolist()
        pairs=dict()
        for idx,feature in enumerate(unique_features):
            pairs[feature]=idx
        return pairs
    def user_item_rating_matrix(self):
        for review in self.reviews:
            user=review["user"]
            item=review["item"]
            rating=review["rating"]
            self.A[user,item]=rating
    def user_feature_attention_matrix(self):
        unique_users=list()
        for review in self.reviews:
            user=review["user"]
            if user not in unique_users:
                unique_users.append(user)
        for user in unique_users:
            feature_frequency=dict() # a dictionary with keys being feature ids and values being feature frequencies 
            for feature in self.feature_index_pair().values():
                feature_frequency[feature]=0 # Initialize the feature frequency dictionary.
            for review in self.reviews:
                if (user!=review["user"]) or ("sentence" not in review.keys()):continue
                feature=review["sentence"][0][0]
                featureid=self.feature_index_pair()[feature]
                feature_frequency[featureid]=feature_frequency[featureid]+1 # Fill the empty feature frequency dictionary with values.
            for idx in feature_frequency.keys():
                if feature_frequency[idx]!=0:
                    feature_frequency[idx]=1+(self.N-1)*(2/(1+np.exp(-feature_frequency[idx]))-1)
            if np.count_nonzero(np.array([value for value in feature_frequency.values()]))!=0:
                for idx in feature_frequency.keys():
                    self.X[user,idx]=int(feature_frequency[idx])/int(np.sum([value for value in feature_frequency.values()]))
    def item_feature_quality_matrix(self):
        unique_items=list()
        for review in self.reviews:
            item=review["item"]
            if item not in unique_items:
                unique_items.append(item)
        for item in unique_items:
            feature_sentiment_pairs=list()
            feature_cumulative_sentiment=dict() # a dictionary with keys being feature ids and values being the cumulative sentiment polarity
            for feature in self.feature_index_pair().values():
                feature_cumulative_sentiment[feature]=0 # Initialize the feature-cumulative sentiment dictionary.
            for review in self.reviews:
                if (item!=review["item"]) or ("sentence" not in review.keys()):continue
                feature=review["sentence"][0][0]
                sentiment=review["sentence"][0][3]
                feature_sentiment_pairs.append((feature,sentiment))
            for feature_sentiment_pair in feature_sentiment_pairs:
                feature=feature_sentiment_pair[0]
                sentiment=feature_sentiment_pair[1]
                featureid=self.feature_index_pair()[feature]
                feature_cumulative_sentiment[featureid]+=sentiment
            for idx in feature_cumulative_sentiment.keys():
                feature_cumulative_sentiment[idx]=1+(self.N-1)/(1+np.exp(-feature_cumulative_sentiment[idx]))
            for idx in feature_cumulative_sentiment.keys():
                self.Y[item,idx]=feature_cumulative_sentiment[idx]/np.sum([value for value in feature_cumulative_sentiment.values()])
               
                