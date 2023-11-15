# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:54:37 2023

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

# Use Tensorboard for network visualization & debugging
from tensorflow.keras.callbacks import TensorBoard


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
    LiDAR_Data['TimeStamp'] = pd.to_datetime(LiDAR_Data['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    
    # Round the microseconds to one decimal place
    LiDAR_Data['TimeStamp'] = LiDAR_Data['TimeStamp'].dt.round('100L')
    
    LiDAR_Data['Year'] = LiDAR_Data['TimeStamp'].dt.year
    
    LiDAR_Data.set_index('TimeStamp', inplace=True)
    LiDAR_Data.sort_index(inplace=True)
    
    print("Original Data has length of ", len(LiDAR_Data))
    
    return LiDAR_Data

def LSTM(data):

    print('Null values: ', data.isnull().sum())
    
    start = data.index[0]
    
    # Find the last non-null timestamp
    end = data.index[-1]
    while pd.isna(end):
        print("not working")
        data = data[:-1]
        end = data.index[-1]
    
    
    missing_time_stamps = pd.date_range(start,
                                        end, 
                                        freq='0.500000min').difference(data.index)
    
    print('Missing time stamps ', missing_time_stamps)
    
    sampling_rate = '0.500000min' # we have 0.500000min resolution in our dataset
    #data_cont = data.resample(sampling_rate).asfreq()
    #data_cont.loc[missing_time_stamps]
    
    data.Wsp_44m.plot(subplots=True, figsize=(20,10), grid=True)
    plt.show()
    
    
    # Choose the input and output features from the main dataframe
    # Note that the index is a datetime object - you might consider converting the dataframe to arrays using df.values
    
    X = data[['W4_Vlos1_orig', 'W4_Vlos2_orig']].values
    Y = data[['Wsp_44m']].values
    
    print(X.shape)
    print(Y.shape)
    
    train_int = int(0.6*len(data)) # 60% of the data length for training
    validation_int = int(0.8*len(data)) # 20% more for validation
    
    # training input vector
    X_train = X[:train_int,:]
    
    # training output vector
    Y_train = Y[:train_int,:]        
    
    # validation input vector
    X_validation = X[train_int:validation_int,:]
    
    # validation output vector
    Y_validation = Y[train_int:validation_int,:]
    
    # test input vector
    X_test = X[validation_int:,:]
    
    # test output vector
    Y_test = Y[validation_int:,:]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)
    
    print(X_train_scaled)
    
    # for multiple model creation - clear  the previous DAG
    K.clear_session() 
    

    # create model - feel free to change the number of neurons per layer
    model = Sequential()
    model.add(Dense(50, 
                    input_dim=X_train_scaled.shape[1], 
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros',
                    activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    now = datetime.now().strftime("%Y%m%d_%H%M")
    
    model_plot_thingie = False
    if model_plot_thingie:
        # Watch for / or \ while creating a directory --> depending on the OS
        tbGraph = TensorBoard(log_dir=f'.\Graph\{now}',
                              histogram_freq=64, write_graph=True, write_images=True)
        
        history = model.fit(X_train_scaled, Y_train, 
                  epochs=30, 
                  batch_size=32,
                  verbose=2,
                  validation_data=(X_validation_scaled, Y_validation),
                  callbacks=[tbGraph])
        
        ### plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()
        
        plt.plot(history.history['binary_accuracy'], label='train_accuracy')
        plt.plot(history.history['val_binary_accuracy'], label='validation_accuracy')
        plt.legend()
        plt.show()
    
    # calculate predictions for validation dataset
    pred_val = model.predict(X_validation_scaled)
    rounded_pred_val = [round(x[0]) for x in pred_val]
    
    plt.figure()
    plt.plot(pred_val,'.', label = 'predictions')
    plt.plot(Y_validation ,'.', label = 'validation dataset') # fill in the validation dataset
    plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
    plt.legend()
    plt.show()
    
    # calculate predictions for test dataset
    pred_test = model.predict(X_test_scaled)
    rounded_pred_test = [round(x[0]) for x in pred_test]
    
    plt.figure()
    plt.plot(pred_test,'.', label = 'predictions')
    plt.plot(Y_test ,'.', label = 'test dataset') # fill in the validation dataset
    plt.plot(rounded_pred_test,'.', label = 'rounded predictions')
    plt.legend()
    plt.show()


# %% MAIN
if __name__ == '__main__':
    data = data_import()
    LSTM(data)