# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:10:49 2021

@author: Administrator
"""

def rounded_to_int(number):
    import numpy as np
    """
    This function rounds the input to its nearest integer value. 
    Parameters
    ----------
    number : input

    Returns
    -------
    nearest_int: the nearest integer to the input

    """
    ceil=np.ceil(number)
    floor=np.floor(number)
    candidates=[ceil,floor]
    differences=[abs(ceil-number),abs(floor-number)]
    nearest_int=candidates[np.argmin(differences)]
    return nearest_int
