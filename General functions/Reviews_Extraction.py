# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 19:09:23 2020

@author: Administrator
"""

def reviews_extraction(user_item_pairs,reviews_parent):
    """
    Parameters
    ----------
    user_item_pairs: a list of user-item index pairs.
    reviews_parent: the parent reviews from which a subset will be extracted based on user_item_pairs. The structure of reviews is a list of dictionaries with the following key-value pairs:
        "user": user_index
        "item": item_index
        "rating": numerical rating
        "text": the textual reviww
        "sentence": the feature-opinion-sentence-polarity quadruple
    Returns
    -------
    reviews_son: the resultant subset. 
    """
    reviews_son=list()
    for pair in user_item_pairs:
        for review in reviews_parent:
            user_item_pair=(review["user"],review["item"])
            if pair==user_item_pair:
                reviews_son.append(review)
    return reviews_son