"""
Created not on Sat Oct 21 13:36:26 2023

@author: not joram
"""
#%% IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.neural_network as nn
import os

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


# %% CUR
cur = os.getcwd()
   
#%% IMPORTING DATA

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

    # print(LiDAR_Data.tail)

    data = LiDAR_Data.dropna()

    print(data.tail)

    return data


#%% PyTorch NEURAL NETWORK

class BasicNN(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(0.1), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(0.1), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=False)


####################################################################################################################################################################################
 # %% NEURAL NETWORK MODEL from Assignment 1

"""
Use this to call this code:

import Surrogate_Model as SM
def train_surrogate_model(AllInputData, AllTargetData):
        
    Xtrain, Xtest, Ytrain, Ytest, Yscaler = SM.scalers(AllInputData, AllTargetData)
    
    return ANNmodel
"""
    
# # SKLEARN Neural Network MLP regressor model

# Y1 = AllTargetData['Blade_root_flapwise_M_x']

# ANNmodel = nn.MLPRegressor()

# ANNmodel.get_params()

# Xscaler = sklearn.preprocessing.StandardScaler()
# Yscaler = sklearn.preprocessing.StandardScaler()
# Xscaler = Xscaler.fit(AllInputData)
# Yscaler = Yscaler.fit(AllTargetData['Blade_root_flapwise_M_x'].values.reshape(-1, 1))

# TrainTestRatio = 0.8
# N = len(AllInputData)
# Xtrain = Xscaler.transform(AllInputData.values[:int(N*TrainTestRatio),:])
# Xtest = Xscaler.transform(AllInputData.values[int(N*TrainTestRatio):,:])

# Ytrain = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[:int(N*TrainTestRatio)].reshape(-1,1))
# Ytest = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[int(N*TrainTestRatio):].reshape(-1,1))


# ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)
# # BEGIN CODE HERE
# ANNmodel.fit(Xtrain, Ytrain.ravel())

# print( 'Train set r-square: ' + str(ANNmodel.score(Xtrain,Ytrain)))
# print( 'Test set r-square: ' + str(ANNmodel.score(Xtest,Ytest)))

# Yout = Yscaler.inverse_transform(ANNmodel.predict(Xtrain).reshape(-1, 1))
# Yout_test = Yscaler.inverse_transform(ANNmodel.predict(Xtest).reshape(-1, 1))

# plt.hist(Yout,50)
# plt.show()

# plt.rc('font', size=14) 
# fig3,axs3 = plt.subplots(1,2,figsize = (16,8))
# plt.setp(axs3[0], title = 'Dependence vs. wind speed', xlabel = 'Mean wind speed [m/s]',ylabel = 'Blade root flapwise moment $M_x$ [Nm]')
# plt.setp(axs3[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
# axs3[0].plot(Xtest[:,0],Yscaler.inverse_transform(Ytest),'o',markersize = 4,color = 'y')
# axs3[0].plot(Xtest[:,0],Yout_test,'*',markersize = 4,color = 'purple')
# axs3[0].legend(['Input data','Model predictions'])
# axs3[1].plot(Yscaler.inverse_transform(Ytest),Yout_test,'ok',markersize = 4)
# axs3[1].plot(np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),\
#                 np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),'-y', linewidth =3)
# axs3[1].legend(['Point-to-point comparisons','1:1 relation'])
# plt.tight_layout()
# plt.savefig(cur + '\\res\\training_1.eps')              
# plt.show()

# plt.rc('font', size=14) 
# fig4,axs4 = plt.subplots(1,2,figsize = (16,8))
# plt.setp(axs4[0], title = 'Dependence vs. Turbulence', xlabel = 'Turbulence [m/s]',ylabel = 'Blade root flapwise moment $M_x$ [Nm]')
# plt.setp(axs4[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
# axs4[0].plot(Xtest[:,1],Yscaler.inverse_transform(Ytest),'o',markersize = 4,color = 'g')
# axs4[0].plot(Xtest[:,1],Yout_test,'*',markersize = 4,color = 'orange')
# axs4[0].legend(['Input data','Model predictions'])
# axs4[1].plot(Yscaler.inverse_transform(Ytest),Yout_test,'ok',markersize = 4)
# axs4[1].plot(np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),\
#                 np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),'-g', linewidth =3)
# axs4[1].legend(['Point-to-point comparisons','1:1 relation'])
# plt.tight_layout() 
# plt.savefig(cur + '\\res\\training_2.eps')             
# plt.show()

# plt.rc('font', size=14) 
# fig5,axs5 = plt.subplots(1,2,figsize = (16,8))
# plt.setp(axs5[0], title = 'Dependence vs. wind shear', xlabel = 'Wind shear exponent [-]',ylabel = 'Blade root flapwise moment $M_x$ [Nm]')
# plt.setp(axs5[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
# axs5[0].plot(Xtest[:,2],Yscaler.inverse_transform(Ytest),'o',markersize = 4,color = 'r')
# axs5[0].plot(Xtest[:,2],Yout_test,'*',markersize = 4,color = 'blue')
# axs5[0].legend(['Input data','Model predictions'])
# axs5[1].plot(Yscaler.inverse_transform(Ytest),Yout_test,'ok',markersize = 4)
# axs5[1].plot(np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),\
#                 np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),'-r', linewidth =3)
# axs5[1].legend(['Point-to-point comparisons','1:1 relation'])
# plt.tight_layout()
# plt.savefig(cur + '\\res\\training_3.eps')              
# plt.show()

# def scalers(AllInputData, AllTargetData):

#     ANNmodel = nn.MLPRegressor()

#     ANNmodel.get_params()

#     Xscaler = sklearn.preprocessing.StandardScaler()
#     Yscaler = sklearn.preprocessing.StandardScaler()
#     Xscaler = Xscaler.fit(AllInputData)
#     Yscaler = Yscaler.fit(AllTargetData['Blade_root_flapwise_M_x'].values.reshape(-1, 1))

#     return Xscaler, Yscaler

# def training(AllInputData, AllTargetData):

#     ANNmodel = nn.MLPRegressor()

#     ANNmodel.get_params()

#     Xscaler = sklearn.preprocessing.StandardScaler()
#     Yscaler = sklearn.preprocessing.StandardScaler()
#     Xscaler = Xscaler.fit(AllInputData)
#     Yscaler = Yscaler.fit(AllTargetData['Blade_root_flapwise_M_x'].values.reshape(-1, 1))

#     TrainTestRatio = 0.8
#     N = len(AllInputData)
#     Xtrain = Xscaler.transform(AllInputData.values[:int(N*TrainTestRatio),:])
#     Xtest = Xscaler.transform(AllInputData.values[int(N*TrainTestRatio):,:])

#     Ytrain = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[:int(N*TrainTestRatio)].reshape(-1,1))
#     Ytest = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[int(N*TrainTestRatio):].reshape(-1,1))


#     ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)
#     # BEGIN CODE HERE
#     ANNmodel.fit(Xtrain, Ytrain.ravel())

#     print( 'Train set r-square: ' + str(ANNmodel.score(Xtrain,Ytrain)))
#     print( 'Test set r-square: ' + str(ANNmodel.score(Xtest,Ytest)))

#     Yout = Yscaler.inverse_transform(ANNmodel.predict(Xtrain).reshape(-1, 1))
#     Yout_test = Yscaler.inverse_transform(ANNmodel.predict(Xtest).reshape(-1, 1))

#     return ANNmodel, Xscaler, Yscaler