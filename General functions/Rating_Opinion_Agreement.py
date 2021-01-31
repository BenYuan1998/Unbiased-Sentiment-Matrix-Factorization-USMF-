# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:42:32 2021

@author: Administrator
"""

def rating_opinion_agreement(df,threshold):
    """

    Parameters
    ----------
    df : the subject dataframe on which a rating feature-based score agreement test will be performed. 
    threshold: the maximum rating feature-based score difference beyond which the numerical rating is judged unfit for representing the ground-truth relevance level for a user-item pair.

    Returns
    -------
    df_qualified: the subset that passes the rating feature-based score agreement test.
    df_disqualified: the subset that fails to pass the rating feature-based score agreement test.
    """
    indices_qualified=list()
    for interaction in df.itertuples():
        rating=getattr(interaction,"rating")
        opin_score=getattr(interaction,"feature_based_score")
        index=getattr(interaction,"Index")
        if abs(rating-opin_score)<=threshold:
            indices_qualified.append(index)
    df_qualified=df[df.index.isin(indices_qualified)]
    df_disqualified=df[~df.index.isin(indices_qualified)]
    return df_qualified,df_disqualified            
            
        