# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:40:56 2023

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

from scipy.stats import truncnorm

from datetime import datetime, timedelta

import warnings

#from Bootstrap import bootstrap_function
#from Joint_Distribution_Fit import Weibull_parameters
#from Surrogate_Model import scalers

import Bootstrap as BS
import Joint_Distribution_Fit as JDF
import Surrogate_Model as SM

# %% FILTER WARNINGS

# Suppress all warnings
warnings.filterwarnings("ignore")

# %% IMPORT DATA

WindData = pd.read_csv('HovsoreData_Sonic_100m_2004-2013.csv')

WindData['Timestamp'] = pd.to_datetime(WindData['Timestamp'], format='%Y%m%d%H%M')

WindData['Year'] = WindData['Timestamp'].dt.year

print("Original Data has length of ", len(WindData))

# %% FILTER
# Filter rows where 'Wsp' is less than or equal to 35 m/s
filtered_WindData = WindData[WindData['Wsp'] <= 35]

filtered_WindData.loc[filtered_WindData["TI"] <= 0.001] = np.nan

filtered_WindData = filtered_WindData.dropna()

# Save the filtered data to a new CSV file
filtered_csv_file = 'FilteredWindData.csv'
filtered_WindData.to_csv(filtered_csv_file, index=False)

WindData = filtered_WindData

print("New Data has length of ", len(WindData))

U = WindData['Wsp']

WindData["SigmaU"] = WindData['Wsp']*WindData['TI']

SigmaU = WindData["SigmaU"]

# %% INPUT DATA and TARGET DATA

InputData = pd.read_excel('ML_ExampleDataSet.xlsx','InputVariables')
#InputData.index = InputData['Sample_No'] # Make the "Sample_No" column as index of the data
InputData = InputData.set_index('Sample_No',drop = False)
InputData # Show the first few rows of the data

TargetData = pd.read_excel('ML_ExampleDataSet.xlsx','LoadResults')
TargetData.set_index('PointNo', drop = False, inplace = True) # Make the "PointNo" column as index of the data
TargetData # Show the first few rows of the data

AllInputData = InputData.where(InputData['Sample_No']==TargetData['PointNo'])
AllTargetData = TargetData.where(TargetData['PointNo']==InputData['Sample_No'])
AllInputData.drop(columns = 'Sample_No', inplace = True)
AllTargetData.drop(columns = 'PointNo', inplace = True)
nsamples = AllInputData['U'].count() # Find the total number of data points in the data frame
FeatureNames = AllInputData.columns.values
DependentVariableNames = AllTargetData.columns.values

Y1 = AllTargetData['Tower_base_fore_aft_M_x']

ANNmodel = nn.MLPRegressor()

ANNmodel.get_params()

#print(AllInputData)
#print(AllTargetData)
AllInputData.drop(columns = 'MannL', inplace = True)
AllInputData.drop(columns = 'MannGamma', inplace = True)
AllInputData.drop(columns = 'VeerDeltaPhi', inplace = True)

#%% U MEAN & U STD

Umean = np.mean(U)
Ustd = np.std(U)

n = len(U) # Count the number of samples


print("Umean = ", Umean)
print("Ustd = ", Ustd)
print("n = ", n)

# %% FUNCTIONS

# Limit state function and gradient function implemented
def limit_state_function(Delta, N, k, Xm, Mx, m):
    g = lambda u: Delta -(1/N*k)*np.sum((Xm*Mx)**m)
    #g_grad = lambda u: 
    return g #, g_grad

def get_alpha(U):
    # Define the parameters
    mean = 0.1
    cov = np.minimum(1, 1 / U)

    # Define the lower and upper bounds for the truncated normal distribution
    lower_bound = 0  # Minimum value
    upper_bound = np.inf  # Positive infinity indicates no upper bound

    # Create the truncated normal distribution
    truncated_normal = truncnorm((lower_bound - mean) / cov, (upper_bound - mean) / cov)

    # Generate random samples from the truncated normal distribution
    sample = truncated_normal.rvs(size=len(U))  # Use the length of the vector 'u'

    return sample

def train_surrogate_model(AllInputData, AllTargetData):
        
    Xtrain, Xtest, Ytrain, Ytest, Yscaler = SM.scalers(AllInputData, AllTargetData)
    
    return Xtrain, Xtest, Ytrain, Ytest, Yscaler

def get_Mx(AllTargetData, input_data, Xscaler, Yscaler):
    
    ANNmodel = nn.MLPRegressor()
    
    #ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)

    Xtrain = Xscaler.transform(input_data)
    
    print(Xtrain.shape)
    
    N_x = len(Xtrain)
    print(N_x)
    
    Ytrain = Yscaler.transform(AllTargetData['Tower_base_fore_aft_M_x'].values.reshape(-1,1))
    
    print(Ytrain.shape)
    
    Xtrain = Xtrain[:Ytrain.shape]
    
    ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)
    
    ANNmodel.fit(Xtrain, Ytrain.ravel())
    
    Mx = Yscaler.inverse_transform(ANNmodel.predict(Xtrain).reshape(-1, 1)).ravel()
    
    return Mx


# %% PRE MADE FUNCTIONS

# CONDITION DISTRIBUTION OF TURBULENCE - BASED ON DATA BINNING
WspBinEdges = np.arange(3.5,33.5,1)
WspBinCenters = WspBinEdges[:-1] + 0.5

MuSigmaBinned = np.zeros(len(WspBinCenters))
SigmaSigmaBinned = np.zeros(len(WspBinCenters))

nData = len(WindData['Wsp'])

# Per wind speed
for iWsp in range(len(WspBinCenters)):
    WspBinSelection = (WindData['Wsp'] > WspBinEdges[iWsp]) & (WindData['Wsp'] <= WspBinEdges[iWsp + 1])
    MuSigmaBinned[iWsp] = np.mean(WindData.loc[WspBinSelection,'SigmaU'])
    SigmaSigmaBinned[iWsp] = np.std(WindData.loc[WspBinSelection,'SigmaU'])
    
Mudatax = WspBinCenters[~np.isnan(MuSigmaBinned)]
Mudatay = MuSigmaBinned[~np.isnan(MuSigmaBinned)]

# Use polyfit (for example np.polyfit). Which order works well - 0, 1, or 2?
pMu = np.polyfit(Mudatax,Mudatay,2)

SigmaSigmaRef = np.mean(SigmaSigmaBinned)
        
MuSigmaFunc = lambda u: pMu[0]*u**2 + pMu[1]*u + pMu[2]

# %% START ROUTINE


# Running Joint Distribution Fit
print("----------------------------------------------------------------------")
print("Getting Weibull Parameters...")
A_weibull, k_weibull = JDF.Weibull_parameters(WindData)
#print("A_Weibull = ", A_weibull)
#print("k_Weibull = ", k_weibull)
print("----------------------------------------------------------------------")


# Running bootstrap
Nbootstrap = 1000 #give number for bootstrap

plots = False # give True if you also wanna see the bootstrap plots

print("----------------------------------------------------------------------")
print("Getting Bootstrap Parameters...")
BootstrapMeans, BootstrapSample, X_w = BS.bootstrap_function(WindData, Nbootstrap, plots)
print("X_w = ", X_w)
print("----------------------------------------------------------------------")

#user_defined = 0.06
delta = [1,0.3]
X_m = [1,0.2]# Loads model uncertainty (you get this from Part3)
#X_W = [1,user_defined]# Uncertainty in wind conditions (you get this from Part2)


# Set Number of Monte Carlo
N_MC = 10**4

k = 4 * 10**12 # Fatigue strength normalization factor (?)
m = 3 # Fatigue S-N curve slope (?)

Delta = stats.lognorm.ppf(np.random.rand(N_MC), s= delta[0], scale = delta[1])
X_M = stats.norm.ppf(np.random.rand(N_MC), loc = X_w[0], scale = X_w[1])
X_W = stats.norm.ppf(np.random.rand(N_MC), loc = X_m[0], scale = X_m[1])

g = np.zeros(X_W.shape)

N_st = 200  #N short term (100-200)

#iterations = N_st

now = datetime.now()

start_time_str = now.strftime("%H:%M:%S")

start_time = datetime.strptime(start_time_str, "%H:%M:%S")

print("Start of Computation Routine: ", start_time_str)
print("======================================================================")

#Xtrain, Xtest, Ytrain, Ytest, Yscaler = train_surrogate_model(AllInputData, AllTargetData)

Xscaler, Yscaler = SM.scalers(AllInputData, AllTargetData)
 
for i in range(N_MC):
    
    print("i = ", i)
    U_routine = stats.weibull_min.ppf(np.random.rand(N_MC), loc = N_st, 
                                      scale = A_weibull*X_W[i], c = k_weibull)
    
    #random.weibull(U, N_st, scale = A_weibull*X_W[i],
    #                           c = k_weibull)
    
    SigmaU_routine = MuSigmaFunc(U_routine)
    
    Alpha_routine = get_alpha(U_routine)
    
    routine_df = pd.DataFrame({'U': U_routine, 
                           'SigmaU': SigmaU_routine,
                           'Alpha' : Alpha_routine
                          })
    
    #Mx_new = insert routine_df in Surrogate model
    M_x = get_Mx(AllTargetData, routine_df, Xscaler, Yscaler)
    
    #M_x = get_Mx(routine_df, AllInputData, AllTargetData)
    
    g[i] = Delta[i] - ( 1 / (N_st*k) ) * sum( (X_M[i]*M_x)**m )
    
    print("current g = ", g[i])
    print("======================================================================")

print("g = ", g)


# %% now calculate probability of failure sum(G<= 0)/NMC
# beta = -normaldist(2,P)

now = datetime.now()

end_time_str = now.strftime("%H:%M:%S")

end_time = datetime.strptime(end_time_str, "%H:%M:%S")
    
print("End of Computation Routine: ", end_time_str)

print("+++++++++++++++++++++++++++++++++++++++++++++")

# =============================================================================
# 
# # Calculate duration
# duration = end_time - start_time
# 
# days, seconds = duration.days, duration.seconds
# hours = days * 24 + seconds // 3600
# minutes = (seconds % 3600) // 60
# seconds = seconds % 60
# 
# print(f"Duration of bootstrap: {hours} hours, {minutes} minutes, {seconds} seconds")
# 
# 
# =============================================================================

PoF = sum(g<=0)/N_MC
beta = JDF.NormalDist(2,PoF)


print("Probability of Failure = ", PoF)
print("Beta of Failure = ", beta)
