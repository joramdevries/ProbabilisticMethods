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

from keras.models import load_model

import LSTM as LM
import FFNN as FM
import understanding as UM
import data_info as DI


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
    
    # Select case
    Beam_lidar_2 = True
    Beam_lidar_4 = True
    control = True
    
    # Select model
    train_FFNN = True
    train_LSTM = True
    show_understanding = False
    show_data_info = True
    
    test_FFNN = False
    test_LSTM = False
    
    #%% MAIN LOOP

    data = data_import()
    if control:
        output = ['MxA1_auto']
        input_data = ['Wsp_44m', 'Wdir_41m']
        model = "control"
    if Beam_lidar_2:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto']
        input_data = ['W2_Vlos1_orig', 'W2_Vlos2_orig']
        model = "lidar2"
    if Beam_lidar_4:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig']
        model = "lidar4"

    if control:
        if train_FFNN:
            FM.FFNN(data, input_data, output,model)
        if train_LSTM:
            LM.LSTM_function(data, input_data, output,model)
        if show_understanding:
            UM.understanding(data, output)
        if show_data_info:
            DI.data_info_plotting(data, output)
            
        if test_FFNN:
            FM.FFNN_testing(data, input_data, output, model)
        if test_LSTM:
            LM.LSTM_testing(data, input_data, output, model)
        
    else:
        if train_FFNN:
            for output in outputs:
                FM.FFNN(data, input_data, output,model)
        if train_LSTM:
            for output in outputs:
                LM.LSTM_function(data, input_data, output,model)
        if show_understanding:
            for output in outputs:
                UM.understanding(data, output)
        if show_data_info:
            for output in outputs:
                DI.data_info_plotting(data, output)
            
        if test_FFNN:
            for output in outputs:
                FM.FFNN_testing(data, input_data, output, model)
        if test_LSTM:
            for output in outputs:
                LM.LSTM_testing(data, input_data, output, model)
        
        
        

    
    