#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:30:00 2020

@author: nuoyuan
"""

def sort_data(data):
    """
    Parameters
    ----------
    x : a dataset stored as a list with the last entry of each element being some date

    Returns
    -------
    x_sorted: the same dataset sorted from the earliest to the latest by date
    """
    import datetime
    for item in data:
        datetime.datetime.strptime(item[-1],"%Y-%m-%d")
        data_sorted=sorted(data,key=lambda x:x[-1],reverse=False)
    return data_sorted
   







