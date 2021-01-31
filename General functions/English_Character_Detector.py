# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:18:15 2020

@author: Administrator
"""

def isEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True
