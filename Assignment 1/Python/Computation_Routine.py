# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:40:56 2023

@author: joram
"""
#%% IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network


# %% FUNCTIONS

# Limit state function and gradient function implemented
def limit_state_function(Delta, N, k, Xm, Mx, m):
    g = lambda u: Delta -(1/N*k)*np.sum((Xm*Mx)**m)
    #g_grad = lambda u: 
    return g #, g_grad

user_defined = 0.2

X_M = [1,0.2]# Loads model uncertainty (you get this from Part3)
X_W = [1,user_defined]# Uncertainty in wind conditions (you get this from Part2)

N = 1000 # User-defined
k = # Fatigue strength normalization factor (?)
m = # Fatigue S-N curve slope (?)