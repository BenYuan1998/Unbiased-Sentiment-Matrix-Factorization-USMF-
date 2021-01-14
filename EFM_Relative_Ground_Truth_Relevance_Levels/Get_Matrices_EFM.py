# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:08:43 2020

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
        for review in self.reviews:
            user=review["user"]
            feature_frequency=dict() # a dictionary with keys being feature ids and values being feature frequencies 
            for review in self.reviews:
                if (user!=review["user"]) or ("sentence" not in review.keys()):continue
                feature=review["sentence"][0][0]
                featureid=self.feature_index_pair()[feature]
                feature_frequency[featureid]=feature_frequency.get(featureid,0)+1
            for idx in feature_frequency.keys():
                if feature_frequency[idx]==0:
                    self.X[user,idx]=0
                else:
                    self.X[user,idx]=1+(self.N-1)*(2/(1+np.exp(-feature_frequency[idx]))-1)
    def item_feature_quality_matrix(self):
        for review in self.reviews:
            item=review["item"]
            feature_sentiment_pairs=list()
            feature_cumulative_sentiment=dict() # a dictionary with keys being feature ids and values being the cumulative sentiment polarity
            for review in self.reviews:
                if (item!=review["item"]) or ("sentence" not in review.keys()):continue
                feature=review["sentence"][0][0]
                sentiment=review["sentence"][0][3]
                feature_sentiment_pairs.append((feature,sentiment))
            for feature in self.feature_index_pair().keys():
                if feature not in [feature_sentiment_pair[0] for feature_sentiment_pair in feature_sentiment_pairs]:continue
                featureid=self.feature_index_pair()[feature]
                feature_sentiment=0
                for feature_sentiment_pair in feature_sentiment_pairs:
                    if feature==feature_sentiment_pair[0]:
                        feature_sentiment+=feature_sentiment_pair[1]
                feature_cumulative_sentiment[featureid]=feature_sentiment
            for idx in feature_cumulative_sentiment.keys():
                if feature_cumulative_sentiment[idx]==0:
                    self.Y[item,idx]=0
                else:
                    self.Y[item,idx]=1+(self.N-1)/(1+np.exp(-feature_cumulative_sentiment[idx])) 








                
                
                
                
            
            
                