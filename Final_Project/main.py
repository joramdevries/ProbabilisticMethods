# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:29:54 2023

@author: joram
"""
#%% IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network as nn

import os

import openturns as ot

from scipy.stats import truncnorm, qmc

from datetime import datetime, timedelta

import warnings


# %% CUR
cur = os.getcwd()

# %% FILTER WARNINGS

# Suppress all warnings
warnings.filterwarnings("ignore")

# %% IMPORT DATA

LiDAR_Data = pd.read_csv('lidar_data_2hz.csv')

LiDAR_Data.drop(columns = 'Unnamed: 0', inplace = True)

LiDAR_Data['TimeStamp'] = pd.to_datetime(LiDAR_Data['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')

LiDAR_Data['Year'] = LiDAR_Data['TimeStamp'].dt.year

print("Original Data has length of ", len(LiDAR_Data))