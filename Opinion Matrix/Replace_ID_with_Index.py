# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:16:07 2020

@author: Administrator
"""

def replace_id_with_index(reviews,userid_to_index,itemid_to_index):
    """
    Parameters
    ----------
    reviews: A list of dictionaries with the following keys:
                'user: user id',
                'itemid: item id',
                'rating',
                'text',
                'sentence' (feature, adj, sent, score)
    userid_to_index : A dictionary each key-value pair of which is an id-index pair for each user.
        DESCRIPTION.
    itemid_to_index : A dictiionary each key-value pair of which is an id-index pair for each item.
    Returns
    -------
    reviews: A list of dictionaries with the following key-value pairs:
                'user: user_index',
                'item: item_index',
                'rating',
                'text',
                'sentence' (feature, adj, sent, score)
    """
    for review in reviews:
        for userid in userid_to_index.keys():
            if review["user"]==userid:
                review["user"]=userid_to_index[userid]
        for itemid in itemid_to_index.keys():
            if review["item"]==itemid:
                review["item"]=itemid_to_index[itemid]
    return reviews
        