# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:34:00 2020

@author: Administrator
"""

from textblob import TextBlob
import pandas as pd

def review_level_opinion_score(df_train):
    """
    Parameters
    ----------
    df_train : the training set stored as a pandas dataframe

    Returns
    -------
    opin_scores : the opinion scores corresponding to all the user textual reviews contained in the training set stored as a Python built-in list
    """
    opin_scores=list()
    for interaction in df_train.itertuples():
        review=getattr(interaction,"review")
        TBobject=TextBlob(review)
        opin_score=TBobject.sentiment.polarity
    # rescale each sentiment polarity score into the same range as that for numerical ratings (i.e., the finite set {1,2,3,4,5})
        if -1<=opin_score<-0.6:
            opin_score_scaled=1
        elif -0.6<=opin_score<-0.2:
            opin_score_scaled=2
        elif -0.2<=opin_score<0.2:
            opin_score_scaled=3
        elif 0.2<=opin_score<0.6:
            opin_score_scaled=4
        else:
            opin_score_scaled=5
        opin_scores.append(opin_score_scaled)
    return opin_scores

