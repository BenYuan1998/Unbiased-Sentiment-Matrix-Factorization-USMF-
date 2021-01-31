#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:06:14 2020

@author: nuoyuan
"""

def ranking_based_on_relevance(list_relevance):
    """
    Parameters
    ----------
    list_relevance : A Python built-in list of relevance levels.

    Returns
    -------
    list_ranking: A Python built-in list of size equal to that of list_rating, 
    containing rank scores computed in a descending fashion (i.e., 
    the largest value in list_rating is assigned a rank score of 1) using the "competition ranking" method 
    (i.e.,  The minimum of the ranks that would have been assigned to all the tied values is assigned to each value.).
    """
    from scipy.stats import rankdata
    list_ranking_reversed=rankdata(list_relevance,method="max")
    list_ranking=list()
    num_elements=len(list_relevance)
    for reversed_rank in list_ranking_reversed:
        list_ranking.append(num_elements-reversed_rank+1)
    return list_ranking
            