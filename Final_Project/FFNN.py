# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:18:15 2023

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


def FFNN(data, output):

    print('Null values: ', data.isnull().sum())
    
    start = data.index[0]
    
    # Find the last non-null timestamp
    end = data.index[-1]
    print(data.tail)
    while pd.isna(end):
        print("not working")
        data = data[:-1]
        end = data.index[-1]
    
    
    missing_time_stamps = pd.date_range(start,
                                        end, 
                                        freq='0.500000min').difference(data.index)
    
    print('Missing time stamps ', missing_time_stamps)
    
    data.Wsp_44m.plot(subplots=True, figsize=(20,10), grid=True)
    plt.show()
    
    
    # Choose the input and output features from the main dataframe
    # Note that the index is a datetime object - you might consider converting the dataframe to arrays using df.values
    loads = np.zeros(len(data['Wsp_44m']))
    data['Loads'] = loads
    
    X = data[['Wsp_44m', 'Wdir_41m']].values
    Y = data[[output]].values
    
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
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)
    
    print(X_train_scaled)
    
    # for multiple model creation - clear  the previous DAG
    K.clear_session() 
    

    # create model - feel free to change the number of neurons per layer
    #model = Sequential()
    #model.add(Dense(50, 
    #                input_dim=X_train_scaled.shape[1], 
    #                kernel_initializer='he_normal',#random_uniform
    #                bias_initializer='zeros',
    #                activation='relu'))
    #model.add(Dense(10, activation='relu'))
    #model.add(Dense(1, activation='linear'))
    #model.summary()
    
    # Create a new model with increased complexity
    model = Sequential()
    model.add(Dense(100, input_dim=X_train_scaled.shape[1], kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    
    model.summary()
        
    def lr_schedule(epoch):
        return 0.001 * 0.9 ** epoch

    # Adjust the learning rate as needed
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=custom_optimizer)
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Add batch normalization and dropout layers as needed
    model.add(BatchNormalization())
    model.add(Dropout(0.125))
    
    #model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    now = datetime.now().strftime("%Y%m%d_%H%M")
    
    model_plot_thingie = True
    if model_plot_thingie:
        # Watch for / or \ while creating a directory --> depending on the OS
        tbGraph = TensorBoard(log_dir=f'.\Graph\{now}',
                              histogram_freq=64, write_graph=True, write_images=True)
        
        history = model.fit(X_train_scaled, Y_train, 
                            epochs=100,  # Increase the number of epochs
                            batch_size=16,
                            verbose=2,
                            validation_data=(X_validation_scaled, Y_validation),
                            callbacks=[tbGraph, early_stopping, lr_scheduler])
                
        ### plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()
        
        #plt.plot(history.history['binary_accuracy'], label='train_accuracy')
        #plt.plot(history.history['val_binary_accuracy'], label='validation_accuracy')
        #plt.legend()
        #plt.show()
    
    # calculate predictions for validation dataset
    pred_val = model.predict(X_validation_scaled)
    rounded_pred_val = [round(x[0]) for x in pred_val]
    
    plt.figure()
    plt.plot(pred_val,'.', label = 'predictions')
    plt.plot(Y_validation ,'.', label = 'validation dataset') # fill in the validation dataset
    #plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
    plt.legend()
    plt.show()
    
    # calculate predictions for test dataset
    pred_test = model.predict(X_test_scaled)
    rounded_pred_test = [round(x[0]) for x in pred_test]
    
    plt.figure()
    plt.plot(pred_test,'.', label = 'predictions')
    plt.plot(Y_test ,'.', label = 'test dataset') # fill in the validation dataset
    #plt.plot(rounded_pred_test,'.', label = 'rounded predictions')
    plt.legend()
    plt.show()
    
    
# %% MAIN
if __name__ == '__main__':
    
    #%% MAIN LOOP

    data = data_import()
    output = 'MxA1_auto'

    FFNN(data, output)
