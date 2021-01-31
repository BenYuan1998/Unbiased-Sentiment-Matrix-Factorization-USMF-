# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:41:57 2021

@author: Administrator
"""


def per_item_interactions(user_item_pairs):
    per_item_interactions=dict() # a dictionary with item indices being the keys and corresponding lists of per-item user interactions being the values 
    for pair in user_item_pairs:
        user=pair[0]
        item=pair[1]
        if item not in per_item_interactions.keys():
            per_item_interactions[item]=list()
        per_item_interactions[item].append(user)
    return per_item_interactions