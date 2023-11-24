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

#from scipy.stats import truncnorm, qmc

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
    #model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu'))
    
    # Third LSTM layer
    #model.add(LSTM(25, activation='relu'))
    
    # Output Layer
    model.add(Dense(1, activation='linear'))
    model.summary()
    
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Add batch normalization and dropout layers as needed
    model.add(BatchNormalization())
    model.add(Dropout(0.125))
    
    # fit the model and store the graphs and performance to be used in TensorBoard (optional)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    
    tbGraph = TensorBoard(log_dir=f'.\Graph\{now}',
                          histogram_freq=64*2, write_graph=True, write_images=True)
    
    history = model.fit(train_X, train_Y, 
              epochs=100,
              batch_size=64,
              verbose=2,
              validation_data=(validation_X, validation_Y),
              callbacks=[tbGraph])
    
    ### plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()
    
    #plt.plot(history.history['binary_accuracy'], label='train_accuracy')
    #plt.plot(history.history['val_binary_accuracy'], label='validation_accuracy')
    #plt.legend()
    #plt.show()
    
    # Save the trained model
    model.save('PMWE_LSTM_Model.h5')
    
    
def LSTM_testing(data, input_data, output):
    
    # Load the model
    model = load_model('PMWE_LSTM_Model.h5')
    
    X = data[[input_data]].values
    Y = data[[output]].values
    
    print(X.shape)
    print(Y.shape)
    
    #train_int = int(0.6*len(data)) # 60% of the data length for training
    validation_int = int(0.8*len(data)) # 20% more for validation
    
    # test input vector
    X_test = X[validation_int:,:]
    
    # test output vector
    Y_test = Y[validation_int:,:]
    
    # Generate predictions on the test set
    test_predictions = model.predict(X_test)
    
    # Flatten the predictions and actual values
    test_predictions_flat = test_predictions.flatten()
    test_actual_values_flat = Y_test.flatten()
    
    # Calculate residuals
    test_residuals = test_actual_values_flat - test_predictions_flat
    
    # Create Q-Q plot using the residuals
    import statsmodels.api as sm
    
    sm.qqplot(test_residuals, line='s')
    plt.title("Q-Q Plot of Test Set Residuals")
    plt.show()

# %% MAIN
if __name__ == '__main__':
    
    #%% CONTROL
    
    training_model = False
    testing_model = True
    
    # Select case
    Beam_lidar_2 = False
    Beam_lidar_4 = False
    control = False
    
    #%% MAIN LOOP

    
    data = data_import()
    
    if control:
        output = 'MxA1_auto'
        input_data = ['Wsp_44m', 'Wdir_41m']
        
        if training_model:
            LSTM_function(data, output)
        if testing_model:
            LSTM_testing(data, input_data, output)
        
    if Beam_lidar_2:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto']
        input_data = ['W2_Vlos1_orig', 'W2_Vlos2_orig']
        
        if training_model:
            
            for output in outputs:
                LSTM_function(data, output)
                
        if testing_model:
            
            for output in outputs:
                LSTM_testing(data, input_data, output)
                
    if Beam_lidar_4:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig']
        
        if training_model:
            
            for output in outputs:
                LSTM_function(data, output)
                
        if testing_model:
            
            for output in outputs:
                LSTM_testing(data, input_data, output)


