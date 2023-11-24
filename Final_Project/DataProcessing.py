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

# from scipy.stats import truncnorm, qmc

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

def data_import(fake_addition=False):
    LiDAR_Data = pd.read_csv('lidar_data_2hz.csv')

    LiDAR_Data.drop(columns='Unnamed: 0', inplace=True)

    # Assuming LiDAR_Data is your DataFrame with a 'TimeStamp' column
    LiDAR_Data['TimeStamp'] = pd.to_datetime(LiDAR_Data['TimeStamp'], format="%Y-%m-%d %H:%M:%S.%f")

    print("Original Data has length of ", len(LiDAR_Data))

    LiDAR_Data.drop_duplicates(subset='TimeStamp')

    LiDAR_Data.set_index('TimeStamp', inplace=True, drop=False)
    LiDAR_Data.dropna(subset=['TimeStamp'], inplace=True)
    LiDAR_Data.drop(columns='TimeStamp', inplace=True)
    LiDAR_Data.sort_index(inplace=True)

    if fake_addition:
        times = LiDAR_Data.index.values
        gaps = times[1:] - times[:-1]

        tracker = True
        ii = 1
        last_start = times[0]
        cutoff_time = 10 #seconds
        cutoff_gap = cutoff_time*2*gaps.min()

        data = []

        while tracker:
            if gaps[ii-1] > cutoff_gap:
                end = LiDAR_Data.index[ii]
                data.append(LiDAR_Data.loc[last_start:end])
                last_start = end
            ii+=1
            if ii == len(times):
                tracker = False
    else:
        data = LiDAR_Data

    # print(data[0])

    # print("New Data has length of ", len(data))
    #
    # # print(data.tail)
    #
    # # Calculate the rolling mean for the last 10 minutes (600 seconds)
    # rolling_mean = data['Wsp_44m'].rolling(window=1200).mean()
    #
    # # Calculate the rolling standard deviation for the last 10 minutes (600 seconds)
    # rolling_std = data['Wsp_44m'].rolling(window=1200).std()
    #
    # # Create a new column 'Mean Ws' with the calculated rolling mean values
    # data['Mean WS'] = rolling_mean
    # data['STD'] = rolling_std
    # data['TI'] = rolling_std / rolling_mean
    #
    # return data


if __name__ == '__main__':
    # %% MAIN LOOP

    data = data_import()