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
    
    
def LSTM_function(data, output):
    
        ### define a function that will prepare the shifting input sequences for the network
    def forecast_sequences_input(input_data,n_lag):
        """
        A function that will split the input time series to sequences for nowcast/forecast problems
        Arguments:
            input_data: Time series of input observations as a list, NumPy array or pandas series
            n_lag: number of previous time steps to use for training, a.k.a. time-lag        
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = input_data.shape[1] 
        df = pd.DataFrame(input_data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_lag, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together (aggregate)
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        return agg


    ### define a function that will prepare the shifting output sequences of the network
    def forecast_sequences_output(output_data,n_out):
        """
        A function that will split the output time series to sequences for nowcast/forecast problems
        Arguments:
            output_data: Time series of input observations as a list, NumPy array or pandas series
            n_out: forecast horizon (for multi-output forecast)
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = output_data.shape[1] 
        df = pd.DataFrame(output_data)
        cols, names = list(), list()
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together (aggregate)
        agg = pd.concat(cols, axis=1)
        agg.columns = names    
        return agg
    
    n_lag = 6 # number of previous time steps to use for training, a.k.a. time-lag
    n_out = 1  # forecast horizon [s] 
    
    ### Split data into train & test 
    
    input1 = 'Wsp_44m'
    input2 = 'Wdir_41m'

    train_int = int(0.6*len(data)) # 60% of the data length for training
    validation_int = int(0.8*len(data)) # 20% more for validation
    
    # training input vector
    X_train = data[[input1, input2]][:train_int]
    X_train = forecast_sequences_input(X_train,n_lag)
    
    # training output vector
    Y_train = data[[output]][:train_int]
    Y_train = forecast_sequences_output(Y_train, n_out)
    
    # validation input vector
    X_validation = data[[input1, input2]][train_int:validation_int]
    X_validation = forecast_sequences_input(X_validation,n_lag)
    
    # validation output vector
    Y_validation = data[[output]][train_int:validation_int]
    Y_validation = forecast_sequences_output(Y_validation, n_out)
    
    # test input vector
    X_test = data[[input1, input2]][validation_int:]
    X_test = forecast_sequences_input(X_test,n_lag)
    
    # test output vector
    Y_test = data[[output]][validation_int:]
    Y_test = forecast_sequences_output(Y_test, n_out)
    
    ### scale the dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)
    
    print('Training input (samples, timesteps):', X_train_scaled.shape)
    print('Training output (samples, timesteps):', Y_train.shape)
    print('Validation input (samples, timesteps):', X_validation_scaled.shape)
    print('Validation output (samples, timesteps):', Y_validation.shape)
    
    
    pad_value = 999

    X_train_scaled[np.isnan(X_train_scaled)] = pad_value
    X_validation_scaled[np.isnan(X_validation_scaled)] = pad_value
    X_test_scaled[np.isnan(X_test_scaled)] = pad_value
    
    # for multiple model creation - clear  the previous DAG
    K.clear_session() 
    
    ### Input reshape for LSTM problem  [samples, timesteps, features]
    no_features = 2 # Avg and Std of wind speed
    
    train_X = X_train_scaled.reshape((X_train_scaled.shape[0], n_lag, no_features))#.astype('float32')
    train_Y = Y_train.values#.astype('float32')
    
    validation_X = X_validation_scaled.reshape((X_validation_scaled.shape[0], n_lag, 2))#.astype('float32')
    validation_Y = Y_validation.values#.astype('float32')
    
    test_X = X_test_scaled.reshape((X_test_scaled.shape[0], n_lag, 2))#.astype('float32')
    test_Y = Y_test.values#.astype('float32')
    
    ### create model
    model = Sequential()
    
    # Masking layer (for the pad_value)
    model.add(Masking(mask_value=pad_value, input_shape=(None, no_features)))
    
    # First LSTM layer
    model.add(LSTM(50, 
                   return_sequences=True,  # important to add it to ensure the following LSTM layers will have the same input shape
                   input_shape=(train_X.shape[1], train_X.shape[2]),                
                   kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
                   
    # then we add the activation
    model.add(Activation('relu'))
    
    # Second LSTM layer
    model.add(LSTM(10, activation='relu'))
    
    # Output Layer
    model.add(Dense(1, activation='linear'))
    model.summary()
    
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # fit the model and store the graphs and performance to be used in TensorBoard (optional)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    
    tbGraph = TensorBoard(log_dir=f'.\Graph\{now}',
                          histogram_freq=64*2, write_graph=True, write_images=True)
    
    history = model.fit(train_X, train_Y, 
              epochs=30,
              batch_size=64,
              verbose=2,
              validation_data=(validation_X, validation_Y),
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
    
    # %% CONTROL BOARD
    
    # Select model
    show_FFNN = False
    show_LSTM = True
    show_understanding = False
    
    #%% MAIN LOOP

    data = data_import()
    output = 'MxA1_auto'
    if show_FFNN:
        FFNN(data, output)
    if show_LSTM:
        LSTM_function(data, output)
    if show_understanding:
        understanding(data, output)