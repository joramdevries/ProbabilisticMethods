# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:23:18 2023

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

def understanding(df, output):
    
    start_timestamp = pd.to_datetime('07/09/2020 13:30:00', format="%d/%m/%Y %H:%M:%S")
    end_timestamp = pd.to_datetime('07/09/2020 13:30:20', format="%d/%m/%Y %H:%M:%S")
    
    filtered = df[(df.index > start_timestamp) & (df.index < end_timestamp)]
    
    print(filtered)
    
    # Create subplots in a single row
    fig, axs = plt.subplots(4, 1, figsize=(15, 8))
    
    # Plot on the first subplot
    axs[0].plot(filtered.index, filtered['ActPow'])
    axs[0].set_ylabel(r'$P [kW]$')
    axs[0].grid(True)
    
    
    # Plot on the second subplot
    axs[1].plot(filtered.index, filtered['MxA1_auto'], 'blue', label = "Blade A")
    axs[1].plot(filtered.index, filtered['MxB1_auto'], 'orange', label = "Blade B")
    axs[1].plot(filtered.index, filtered['MxC1_auto'], 'green', label = "Blade C")
    axs[1].set_ylabel(r'$M_{bf} [kNm]$')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot on the third subplot
    axs[2].plot(filtered.index, filtered['W2_Vlos1_orig'] , 'blue', label = "Vlos1")
    axs[2].plot(filtered.index, filtered['W2_Vlos2_orig'] , 'orange', label = "Vlos2")
    axs[2].set_ylabel(r'$U [m/s]$')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot on the fourth subplot
    axs[3].plot(filtered.index, filtered['W4_Vlos1_orig'] , 'blue', label = "Vlos1")
    axs[3].plot(filtered.index, filtered['W4_Vlos2_orig'] , 'orange', label = "Vlos2")
    axs[3].plot(filtered.index, filtered['W4_Vlos3_orig'] , 'green', label = "Vlos3")
    axs[3].plot(filtered.index, filtered['W4_Vlos4_orig'] , 'red', label = "Vlos4")
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel(r'$U [m/s]$')
    axs[3].legend()
    axs[3].grid(True)
    
    # Add legends after all lines have been plotted
    #axs[1].legend(loc='bottom right')
    #axs[2].legend(loc='bottom right')
    #axs[3].legend(loc='bottom right')
    
    # Adjust layout for better appearance
    plt.tight_layout()
    
    # Show the plots
    plt.show()
    
    
    print('lenght of df =',len(df))
    df = df[df['Mean WS']>4]
    print('lenght of df =',len(df))
    #df = data[265<data['Wdir_41m']<295]
    print('lenght of df =',len(df))
    data = df[0<df['ActPow']]
    print('lenght of df =',len(data))
    df = data[16<data['ROT']]
    print('lenght of df =',len(df))
    data = df[df['Pitch']<23]
    print('lenght of df =',len(data))
    
    df = data.dropna()
    
    wind_speed_data = df['Mean WS']
    ti_data = df['TI']
    
    # Create a 2D histogram
    hist, x_edges, y_edges = np.histogram2d(wind_speed_data, ti_data, bins=(20, 20))
    
    # Create a meshgrid for the heatmap
    x, y = np.meshgrid(x_edges[:-1], y_edges[:-1])
    
    # Plot the 2D histogram as a heatmap
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x, y, hist.T, cmap='viridis')
    plt.colorbar(label='Count')
    
    # Add labels and title
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Turbulence Intensity')
    plt.title('Wind Speed vs Turbulence Intensity Distribution')
    
    # Show the plot
    plt.grid(True)
    plt.show()
    
    
# %% MAIN
if __name__ == '__main__':
    
    #%% MAIN LOOP

    data = data_import()
    output = 'MxA1_auto'

    understanding(data, output)
    