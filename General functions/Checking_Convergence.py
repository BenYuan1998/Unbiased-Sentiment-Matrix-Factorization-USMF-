# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:33:29 2021

@author: Administrator
"""

def checking_convergence(iterations,costs):
    import matplotlib.pyplot as plt
    plt.plot(iterations,costs)
    plt.xlabel("ith iteration")
    plt.ylabel("value of the loss function")
    plt.show()