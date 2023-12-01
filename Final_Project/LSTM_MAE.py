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

from sklearn.metrics import mean_absolute_error

import glob


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


    
    
def LSTM_function(data, input_data, output, model_name):
    
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
    
    #input1 = 'Wsp_44m'
    #input2 = 'Wdir_41m'

    train_int = int(0.6*len(data)) # 60% of the data length for training
    validation_int = int(0.8*len(data)) # 20% more for validation
    
    Y = data[output]
    # training input vector
    X_train = data[input_data][:train_int]
    X_train = forecast_sequences_input(X_train,n_lag)
    
    # training output vector
    Y_train = Y[:train_int]
    Y_train = forecast_sequences_output(Y_train, n_out)
    
    # validation input vector
    X_validation = data[input_data][train_int:validation_int]
    X_validation = forecast_sequences_input(X_validation,n_lag)
    
    # validation output vector
    Y_validation = Y[train_int:validation_int]
    Y_validation = forecast_sequences_output(Y_validation, n_out)
    
    # test input vector
    X_test = data[input_data][validation_int:]
    X_test = forecast_sequences_input(X_test,n_lag)
    
    # test output vector
    Y_test = Y[validation_int:]
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
    no_features = len(input_data) #2 # Avg and Std of wind speed
    
    train_X = X_train_scaled.reshape((X_train_scaled.shape[0], n_lag, no_features))#.astype('float32')
    train_Y = Y_train.values#.astype('float32')
    
    validation_X = X_validation_scaled.reshape((X_validation_scaled.shape[0], n_lag, no_features))#.astype('float32')
    validation_Y = Y_validation.values#.astype('float32')
    
    test_X = X_test_scaled.reshape((X_test_scaled.shape[0], n_lag, no_features))#.astype('float32')
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
      
    if model_name == 'lidar4moredata_RELU_mae':
        model.add(Activation('relu'))
        model.add(LSTM(10, activation='relu'))
        
    else:
        model.add(Activation('tanh'))
        
        # Second LSTM layer
        #model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(10, activation='tanh'))
             
    # then we add the activation
    #model.add(Activation('tanh'))
    
    # Second LSTM layer
    #model.add(LSTM(50, activation='relu', return_sequences=True))
    #model.add(LSTM(10, activation='tanh', return_sequences=True))
    
    # Third LSTM layer
    #model.add(LSTM(25, activation='tanh'))
    
    # Third LSTM layer (new layer)
    #model.add(LSTM(25, activation='tanh', return_sequences=True))

    # Fourth LSTM layer (new layer)
    #model.add(LSTM(10, activation='tanh'))
    
    # Output Layer
    model.add(Dense(len(output), activation='linear'))
    model.summary()
    
    def lr_schedule(epoch):
        return 0.01 * 0.9 ** epoch
    
    # compile the model
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
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
              callbacks=[tbGraph, early_stopping, lr_scheduler])
    
    ### plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    
    plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{output}_loss.eps')
    plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{output}_loss.jpg')
    plt.show()
    
    #plt.plot(history.history['binary_accuracy'], label='train_accuracy')
    #plt.plot(history.history['val_binary_accuracy'], label='validation_accuracy')
    #plt.legend()
    #plt.show()
    
    # Save the trained model
    model.save(f'PMWE_LSTM_mae_Model_{model_name}_{output}.h5')
    
    
def LSTM_testing(data, input_data, outputs, model_name):
    
    # Load the model
    model = load_model(f'PMWE_LSTM_mae_Model_{model_name}_{outputs}.h5')
    
    print("outputs:",outputs)
    #print("output:",output)
    X = data[input_data].values
    Y = data[outputs].values
    
    X_data = data[input_data]
    Y_data = data[outputs]
    
    print(X.shape)
    print(Y.shape)
    
    train_int = int(0.6 * len(data))  # 60% of the data length for training
    validation_int = int(0.8 * len(data))  # 20% more for validation

    # training input vector
    X_train = X[:train_int, :]

    # training output vector
    Y_train = Y[:train_int, :]

    # validation input vector
    X_validation = X[train_int:validation_int, :]

    # validation output vector
    Y_validation = Y[train_int:validation_int, :]

    # test input vector
    X_test = X[validation_int:, :]

    # test output vector
    Y_test = Y[validation_int:, :]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)
    
    X_validation_scaled_reshaped = X_validation_scaled.reshape((X_validation_scaled.shape[0], 1, X_validation_scaled.shape[1]))
    
    
    #train_int = int(0.6*len(data)) # 60% of the data length for training
    #validation_int = int(0.8*len(data)) # 20% more for validation
    
    # test input vector
    #X_test = X[validation_int:,:]
    
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # test output vector
    #Y_test = Y[validation_int:,:]
    
    # Generate predictions on the test set
    test_predictions = model.predict(X_test_reshaped)
    
    # Flatten the predictions and actual values
    test_predictions_flat = test_predictions.flatten()
    test_actual_values_flat = Y_test.flatten()
    
    # Calculate residuals
    test_residuals = test_actual_values_flat - test_predictions_flat
    
    # Create Q-Q plot using the residuals
    import statsmodels.api as sm
    
    sm.qqplot(test_residuals, line='s')
    plt.title("Q-Q Plot of Test Set Residuals")
    
    plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{outputs}_QQ.eps')
    plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{outputs}_QQ.jpg')
    
    plt.show()
    
    plt.figure()
    plt.scatter(test_actual_values_flat,test_predictions_flat, marker='.')
    plt.xlabel(f"Actual Values {outputs}")
    plt.ylabel(f"Predicted Values {outputs}")
    
    plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{outputs}_Predict_vs_Actual.eps')
    plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{outputs}_Predict_vs_Actual.jpg')
    
    plt.show()
    

    # calculate predictions for validation dataset
    pred_val = model.predict(X_validation_scaled_reshaped)
    rounded_pred_val = [round(x[0]) for x in pred_val]

    for i in range(len(Y_validation[0])):
        plt.figure()
        plt.plot(Y_validation[:, i], '.', label='validation dataset')  # fill in the validation dataset
        plt.plot(pred_val[:, i], '.', label=Y_data.columns[i]+' predictions')
        # plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
        
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions.eps')
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions.jpg')
        
        plt.legend()
        
        plt.figure()
        plt.scatter(Y_validation[:, i], pred_val[:, i])  # fill in the validation dataset
        plt.xlabel(f"Validation {Y_data.columns[i]}")
        plt.ylabel(f"Prediction {Y_data.columns[i]}")
        # plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
        
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions_2.eps')
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions_2.jpg')
        
        plt.legend()
        
        plt.figure()
        plt.plot(Y_validation[:, i], '.', label='validation dataset')  # fill in the validation dataset
        plt.plot(pred_val[:, i], '.', label=Y_data.columns[i]+' predictions')
        # plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
        plt.xlim([0,600])
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions_xlim600.eps')
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions_xlim600.jpg')
        
        plt.legend()
        
        num_offsets = 20
        mae_values = []
        
        plt.figure()
        for j in range(num_offsets):
            offset = 2 * j
            mae = mean_absolute_error(Y_validation[100:1000, i], pred_val[100 - offset:1000 - offset, i])
            mae_values.append(mae)
            print(f'MAE for offset {j}s: {mae}')
            
            # Plot validation dataset for each offset if j is even
            if j % 2 == 0 and j < 14:
                  # fill in the validation dataset
                plt.plot(pred_val[100 - offset:1000 - offset, i], '-', label=Y_data.columns[i]+f' predictions (Offset: {j}s)')
            
            # plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
        plt.plot(Y_validation[100:1000, i], '.', label='validation dataset')
        plt.xlim([100,600])
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions_xlim600_offset.eps')
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_validation_predictions_xlim600_offset.jpg')
            
        plt.legend()

        #print(f'Mean Absolute Error for {Y_data.columns[i]}: {mae}')
        
    plt.show()
    
    # If you want to save the MAE values for later analysis
    mae_dict = dict(zip(range(num_offsets), mae_values))
    
    # Convert the dictionary to a DataFrame
    mae_df = pd.DataFrame(list(mae_dict.items()), columns=['Offset', 'MAE'])
    
    # Save the DataFrame to a CSV file
    mae_df.to_csv(f'CSV/mae_results_{model_name}_mae.csv', index=False)

    # calculate predictions for test dataset
    pred_test = model.predict(X_test_reshaped)
    rounded_pred_test = [round(x[0]) for x in pred_test]

    for i in range(len(Y_validation[0])):
        plt.figure()
        plt.plot(Y_test[:, i], '.', label='test dataset')  # fill in the validation dataset
        plt.plot(pred_test[:, i], '.', label=Y_data.columns[i] + ' predictions')
        # plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
        
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_test_predictions.eps')
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_test_predictions.jpg')
        
        plt.legend()
        
        plt.figure()
        plt.plot(Y_test[:, i], '.', label='test dataset')  # fill in the validation dataset
        plt.plot(pred_test[:, i], '.', label=Y_data.columns[i] + ' predictions')
        # plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
        plt.xlim([0,600])
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_test_predictions_xlim600.eps')
        plt.savefig(f'Plots/PMWE_LSTM_mae_Model_{model_name}_{Y_data.columns[i]}_test_predictions_xlim600.jpg')
        
        plt.legend()

    plt.show()
    for i in range(len(Y_validation[0])):
        y_label_for_u = f'{Y_data.columns[i]}'
     
        plt.rc('font', size=14) 
        fig2a,axs2a = plt.subplots(1,2,figsize = (16,8))
        plt.setp(axs2a[0], title = 'Dependence vs. wind speed', xlabel = 'Mean wind speed [m/s]',ylabel = y_label_for_u)
        plt.setp(axs2a[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
        axs2a[0].plot(X_test[:,0],Y_test[:, i],'o',markersize = 4,color = 'y')
        axs2a[0].plot(X_test[:,0],pred_test[:, i],'x',markersize = 4,color = 'purple')
        axs2a[0].legend(['Input data','Model predictions'])
        axs2a[1].plot(Y_test[:, i],pred_test[:, i],'ok',markersize = 4)
        axs2a[1].plot(np.array([np.min(Y_test[:, i]), np.max(Y_test[:, i])]),\
                     np.array([np.min(Y_test[:, i]), np.max(Y_test[:, i])]),'-y',linewidth = 4)
        axs2a[1].legend(['Point-to-point comparisons','1:1 relation'])
        plt.tight_layout()  
        plt.savefig(f'Plots/.Assignment1_graph_{Y_data.columns[i]}_{model_name}.eps') 
        plt.savefig(f'Plots/.Assignment1_graph_{Y_data.columns[i]}_{model_name}.jpg')          
        plt.show()
        
    

# %% MAIN
if __name__ == '__main__':
    
    
    #%% CONTROL
    
    training_model = True
    testing_model = True
    
    # Select case
    Beam_lidar_2 = False
    Beam_lidar_4 = False
    control = False
    
    Beam_lidar_2_plus_turbine = False
    Beam_lidar_2_more_data = False
    Beam_lidar_4_plus_turbine = False
    Beam_lidar_4_more_data = False
    
    Beam_lidar_2_batch1024 = False #doenst work well
    Beam_lidar_4_batch1024 = False #doesnt work well
    
    
    #Mean Absolute Error
    mae_plot = False
    
    
    #Extra LSTM layer
    layered = False
    
    
    #learning rate
    learning_Rate =False
    
    
    #relu
    relu = False
    
    #only power
    power_only = False
    
    #only loads
    loads_only = True
    
    #%% MAIN LOOP

    
    data = data_import()
    
    if control:
        output = ['MxA1_auto']
        input_data = ['Wsp_44m', 'Wdir_41m']
        model = "control"
        
        if training_model:
            LSTM_function(data, input_data, output, model)
        if testing_model:
            LSTM_testing(data, input_data, output, model)
        
    if Beam_lidar_2:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto']
        input_data = ['W2_Vlos1_orig', 'W2_Vlos2_orig']
        model = "lidar2"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
                
    if Beam_lidar_4:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig']
        model = "lidar4"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
    if Beam_lidar_2_more_data:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto','ActPow']
        input_data = ['W2_Vlos1_orig', 'W2_Vlos2_orig','W2_phi','u2',
                      'v2','U2','phi2']
        model = "lidar2moredata"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
                
    if Beam_lidar_4_more_data:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto','ActPow']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig','W4_phi','u4_top',
                      'v4_top','U4_top','phi4_top','u4_bot','v4_bot','U4_bot','phi4_bot']
        model = "lidar4moredata"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
    if power_only:
        outputs = ['ActPow']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig','W4_phi','u4_top',
                      'v4_top','U4_top','phi4_top','u4_bot','v4_bot','U4_bot','phi4_bot']
        model = "lidar4_onlypower_tanh"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
    if loads_only:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig','W4_phi','u4_top',
                      'v4_top','U4_top','phi4_top','u4_bot','v4_bot','U4_bot','phi4_bot']
        model = "lidar4_onlyloads"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
    if layered:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto','ActPow']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig','W4_phi','u4_top',
                      'v4_top','U4_top','phi4_top','u4_bot','v4_bot','U4_bot','phi4_bot']
        model = "lidar4moredata_layered"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
    if relu:
        print(data['MxA1_auto'])
        data['MxA1_auto'] = -data['MxA1_auto']
        print(data['MxA1_auto'])
        data['MxB1_auto'] = -data['MxB1_auto']
        data['MxC1_auto'] = -data['MxC1_auto']
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto','ActPow']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig','W4_phi','u4_top',
                      'v4_top','U4_top','phi4_top','u4_bot','v4_bot','U4_bot','phi4_bot']
        model = "lidar4moredata_RELU"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
    if learning_Rate:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto','ActPow']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig','W4_phi','u4_top',
                      'v4_top','U4_top','phi4_top','u4_bot','v4_bot','U4_bot','phi4_bot']
        model = "lidar4moredata_learningrate"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
            
    if Beam_lidar_2_batch1024:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto','ActPow']
        input_data = ['W2_Vlos1_orig', 'W2_Vlos2_orig','W2_phi','u2',
                      'v2','U2','phi2']
        model = "lidar2_batch1204"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
                
    if Beam_lidar_4_batch1024:
        outputs = ['MxA1_auto','MxB1_auto','MxC1_auto','ActPow']
        input_data = ['W4_Vlos1_orig', 'W4_Vlos2_orig','W4_Vlos3_orig','W4_Vlos4_orig','W4_phi','u4_top',
                      'v4_top','U4_top','phi4_top','u4_bot','v4_bot','U4_bot','phi4_bot']
        model = "lidar4_batch1204"
        
        if training_model:
            LSTM_function(data, input_data, outputs,model)
                
        if testing_model:
            LSTM_testing(data, input_data, outputs,model)
            
            
    if mae_plot:
        
        # Assume you have CSV files named 'mae_results_model1.csv', 'mae_results_model2.csv', etc.
        csv_files = glob.glob('CSV/mae_results_*.csv')  # Modify the pattern based on your filenames
        
        plt.figure()
        
        for csv_file in csv_files:
            # Read the CSV file into a DataFrame
            mae_df = pd.read_csv(csv_file)
        
            # Extract model name from the CSV filename
            model_name = csv_file.replace('mae_results_', '').replace('.csv', '')
        
            # Plot MAE against offsets for each model
            plt.plot(mae_df['Offset'], mae_df['MAE'], marker='o', label=model_name)
        
        plt.title('MAE vs Offsets for Different Models')
        plt.xlabel('Offset')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True)
        plt.show()


