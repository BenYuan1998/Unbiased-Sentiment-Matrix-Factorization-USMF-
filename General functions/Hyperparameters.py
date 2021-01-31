#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:04:26 2020

@author: nuoyuan
"""

def hyperparameter_combinations(param_grid):
    """
    Description: This function returns all the hyperparameter combinations whose elements are drawn from the parameter grids

    Parameters
    ----------
    param_grid : A python built-in dictionary with each key being the name of some hyperparameter and its corresponding value being a list of possible values organized in ascending order
                 Example: {"regularization coefficient:[0.2,0.4,0.8,1.0],
                           "learning rate": [0.05,0.1,0.15,0.2]
                           "number of latent factors":[20,40,60,80,100,120]}

    Returns
    -------
    param_combinations: A numpy array with all the hyperparameter combinations stored as rows and the index of each hyperparameter in the rows is the same as the index of that hyperparameter in a list made up of the names of the hyperparameters 

    """
    import itertools
    import numpy as np
    keys,values=zip(*param_grid.items())
    for iteration,value in enumerate(itertools.product(*values)):
        if iteration==0:
            param_combinations=np.array(value)
        else:
            param_combinations=np.vstack((param_combinations,value))
    return param_combinations