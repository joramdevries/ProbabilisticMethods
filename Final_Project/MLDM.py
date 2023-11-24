# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:18:15 2023

@author: joram
"""

# %% IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network as nn
from sklearn.preprocessing import MinMaxScaler
from DataProcessing import data_import

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

# %% CUR
cur = os.getcwd()

# %% FILTER WARNINGS

# Suppress all warnings
warnings.filterwarnings("ignore")


# %% IMPORT DATA


def FFNN(data, output):
    print('Null values: ', data.isnull().sum())
    # track = False
    # while not track:
    #     i = 0
    #     # print(data.iloc[i])
    #     if np.all(data.iloc[i].isnull()):
    #         nullrow = data.loc[i]
    #         track = True
    #     else:
    #         i+=1
    #     if i == len(data):
    #         raise ValueError("Fuck")

    start = data.index[0]
    # while pd.isna(start):
    #     # print("not working")
    #     data = data[1:]
    #     start = data.index[0]

    # Find the last non-null timestamp
    end = data.index[-1]
    # print(data.tail)
    # while pd.isna(end):
    #     # print("not working")
    #     data = data[:-1]
    #     end = data.index[-1]
    print(start, end)

    missing_time_stamps = pd.date_range(start,
                                        end,
                                        freq='0.500000S').difference(data.index)

    # print('Missing time stamps ', missing_time_stamps, "\nTotal: ", len(missing_time_stamps))

    # print(data.loc[missing_time_stamps[0]])
    # for t in missing_time_stamps:
    # emptydata = pd.DataFrame(data=pd.concat([nullrow]*len(missing_time_stamps)), index=missing_time_stamps)
    # data = pd.merge(data, emptydata, how='outer')

    # print(data.loc[missing_time_stamps])


    data.Wsp_44m.plot(subplots=True, figsize=(20, 10), grid=True)
    # plt.show()

    # print(data.columns)

    X0 = data[['W4_Vlos1_orig', 'W4_Vlos2_orig', 'W4_Vlos3_orig', 'W4_Vlos4_orig', 'W2_Vlos1_orig', 'W2_Vlos2_orig']]
    Y0 = data[['MxA1_auto', 'MxB1_auto', 'MxC1_auto']]

    hist_steps = 5
    pred_steps = 3

    # print("check: ", X0.index[0])
    X = X0.copy()
    Y = Y0.copy()


    for j in range(1, hist_steps+1):
        Xj = X0.copy()

        times = X0.index
        newtimes = times - j * timedelta(milliseconds=500)
        Xj.set_index(newtimes, inplace=True)

        col_names = list(X0.columns)
        newcolumns = []
        for i in range(len(col_names)):
            newcolumns.append(col_names[i] + "_t-" + str(j))
            Xj.rename(columns={col_names[i]: newcolumns[i]}, inplace=True)

        X[newcolumns] = list(np.ones((len(times),len(newcolumns)))*np.nan)

        X.combine_first(Xj)






    # Choose the input and output features from the main dataframe
    # Note that the index is a datetime object - you might consider converting the dataframe to arrays using df.values
    loads = np.zeros(len(data['Wsp_44m']))
    data['Loads'] = loads



    X = data[['Wsp_44m', 'Wdir_41m']].values
    Y = data[[output]].values

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

    print(X_train_scaled)

    # for multiple model creation - clear  the previous DAG
    K.clear_session()

    # create model - feel free to change the number of neurons per layer
    # model = Sequential()
    # model.add(Dense(50,
    #                input_dim=X_train_scaled.shape[1],
    #                kernel_initializer='he_normal',#random_uniform
    #                bias_initializer='zeros',
    #                activation='relu'))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    # model.summary()

    # Create a new model with increased complexity
    model = Sequential()
    model.add(
        Dense(100, input_dim=X_train_scaled.shape[1], kernel_initializer='random_uniform', bias_initializer='zeros',
              activation='relu'))
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

    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

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

        # plt.plot(history.history['binary_accuracy'], label='train_accuracy')
        # plt.plot(history.history['val_binary_accuracy'], label='validation_accuracy')
        # plt.legend()
        # plt.show()

    # calculate predictions for validation dataset
    pred_val = model.predict(X_validation_scaled)
    rounded_pred_val = [round(x[0]) for x in pred_val]

    plt.figure()
    plt.plot(pred_val, '.', label='predictions')
    plt.plot(Y_validation, '.', label='validation dataset')  # fill in the validation dataset
    # plt.plot(rounded_pred_val,'.', label = 'rounded predictions')
    plt.legend()
    plt.show()

    # calculate predictions for test dataset
    pred_test = model.predict(X_test_scaled)
    rounded_pred_test = [round(x[0]) for x in pred_test]

    plt.figure()
    plt.plot(pred_test, '.', label='predictions')
    plt.plot(Y_test, '.', label='test dataset')  # fill in the validation dataset
    # plt.plot(rounded_pred_test,'.', label = 'rounded predictions')
    plt.legend()
    plt.show()


# %% MAIN
if __name__ == '__main__':
    # %% MAIN LOOP

    data = data_import()
    output = 'MxA1_auto'

    FFNN(data, output)
