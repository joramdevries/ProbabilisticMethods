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
from sklearn.preprocessing import MinMaxScaler

import os

import openturns as ot

from scipy.stats import truncnorm, qmc

from datetime import datetime, timedelta

import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import LSTM, Masking

# Use Tensorboard for network visualization & debugging
from tensorflow.keras.callbacks import TensorBoard

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import BatchNormalization, Dropout

import LSTM as LM
import FFNN as FM
import understanding as UM


# %% CUR
cur = os.getcwd()

# %% FILTER WARNINGS

# Suppress all warnings
warnings.filterwarnings("ignore")

# %% IMPORT DATA

def data_import():

    LiDAR_Data = pd.read_csv('lidar_data_2hz.csv')
    
    LiDAR_Data.drop(columns = 'Unnamed: 0', inplace = True)
    
    # Assuming LiDAR_Data is your DataFrame with a 'TimeStamp' column
    LiDAR_Data['TimeStamp'] = pd.to_datetime(LiDAR_Data['TimeStamp'], format="%Y-%m-%d %H:%M:%S.%f")
    
    # Round the microseconds to one decimal place
    # LiDAR_Data['TimeStamp'] = LiDAR_Data['TimeStamp'].dt.round('100L')
    
    #LiDAR_Data['Year'] = LiDAR_Data['TimeStamp'].dt.year
    
    LiDAR_Data.set_index('TimeStamp', inplace=True)
    LiDAR_Data.sort_index(inplace=True)
    
    print("Original Data has length of ", len(LiDAR_Data))
    
    print(LiDAR_Data.tail)
    
    data = LiDAR_Data.dropna()
    
    print(data.tail)
    
    # Calculate the rolling mean for the last 10 minutes (600 seconds)
    rolling_mean = data['Wsp_44m'].rolling(window=1200).mean()
    
    # Calculate the rolling standard deviation for the last 10 minutes (600 seconds)
    rolling_std = data['Wsp_44m'].rolling(window=1200).std()
    
    # Create a new column 'Mean Ws' with the calculated rolling mean values
    data['Mean WS'] = rolling_mean
    data['STD'] = rolling_std
    data['TI'] = rolling_std/rolling_mean
        
    return data

# %% MAIN
if __name__ == '__main__':
    
    # %% CONTROL BOARD
    
    # Select model
    show_FFNN = False
    show_LSTM = False
    show_understanding = False
    
    #%% MAIN LOOP

    data = data_import()
    output = 'MxA1_auto'
    if show_FFNN:
        FM.FFNN(data, output)
    if show_LSTM:
        LM.LSTM_function(data, output)
    if show_understanding:
        UM.understanding(data, output)
    