# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:37:22 2023

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

import seaborn as sns

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

def data_info_plotting(df, output):
    
    # Assuming df is your DataFrame
    df['Mean WS'] = pd.to_numeric(df['Mean WS'], errors='coerce')
    df['ActPow'] = pd.to_numeric(df['ActPow'], errors='coerce')
    
    df = df.dropna(subset=['Mean WS', 'ActPow'])
    
    avg_power = df.groupby('Mean WS')['ActPow'].mean().reset_index()
    
    plt.figure()
    plt.scatter(df['Mean WS'], df['ActPow'], marker='.', label="Power")
    #plt.plot(avg_power['Mean WS'], avg_power['ActPow'], color='red', linestyle='-', linewidth=2, label='Average Power')
    plt.legend()
    plt.xlabel('Mean WS')
    plt.ylabel('ActPow')
    plt.title('Scatter Plot with Average Power Curve')
    plt.show()
    
    
def correlation(df, output):
    
    print("Start correlation.....")
    
    # Generate a correlation matrix
    correlation_matrix = df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap using seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    
    # Show the plot
    plt.show()
    
    # Select specific columns for the pairplot
    columns_to_plot = ["Yaw", "ROT", "Wsp_44m", "Wdir_41m", "Pitch", "Azzimuth"]
    
    # Create a pairplot
    sns.pairplot(df[columns_to_plot])
    
    # Show the plot
    plt.show()
    
    # Select specific columns for the pairplot
    columns_to_plot_2 = ["W4_Vlos1_orig", "W4_Vlos2_orig", "W4_Vlos3_orig", "W4_Vlos4_orig", "W2_Vlos1_orig", "W2_Vlos2_orig"]
    
    # Create a pairplot
    sns.pairplot(df[columns_to_plot_2])
    
    # Show the plot
    plt.show()
    
    
def power_plots(df, output):
    
    print("Power plots")
    
# %% MAIN
if __name__ == '__main__':
    
    #%% MAIN LOOP
    correlation = False

    data = data_import()
    output = 'MxA1_auto'

    if correlation:
        correlation(data, output)
        
        
    data_info_plotting(data, output)
    power_plots(data, output)
    
    
    
    