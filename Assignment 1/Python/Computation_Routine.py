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

import os

import openturns as ot

from scipy.stats import truncnorm, qmc

from datetime import datetime, timedelta

import warnings

#from Bootstrap import bootstrap_function
#from Joint_Distribution_Fit import Weibull_parameters
#from Surrogate_Model import scalers

import Bootstrap as BS
import Joint_Distribution_Fit as JDF
import Surrogate_Model as SM

# %% CUR
cur = os.getcwd()

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

Y1 = AllTargetData['Blade_root_flapwise_M_x']

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
    
    return ANNmodel

def get_Mx(AllTargetData, input_data, Xscaler, Yscaler, ANNmodel):
    
    #ANNmodel = nn.MLPRegressor()
    
    #ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)

    #Xtrain = Xscaler.transform(input_data)
    
    #print(Xtrain.shape)
    
    #N_x = len(Xtrain)
    #print(N_x)
    
    #Ytrain = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values.reshape(-1,1))
    
    #print(Ytrain.shape)
    
    #Xtrain = Xtrain[:Ytrain.shape]
    
    #ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)
    
    #ANNmodel.fit(Xtrain, Ytrain.ravel())
    
    #np.atleast_2d
    
    X_scaled = Xscaler.transform(input_data)
    
    Mx_scaled = ANNmodel.predict(X_scaled)
    
    Mx = Yscaler.inverse_transform(Mx_scaled.reshape(-1, 1)).ravel()
    
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
SigmaSigmaFunc = lambda u: SigmaSigmaRef*np.ones(len(u))

SigmaU = lambda u, F: JDF.LogNormDist(2,F,MuSigmaFunc(u),SigmaSigmaFunc(u))


MuAlphaFunc = lambda u: 0.1*np.ones(len(u))
SigmaAlphaFunc = lambda u: np.min(np.concatenate([[np.ones(len(u))],[1/u]]), axis=0)

Alpha = lambda u, F: JDF.NormalDist(2,F,MuAlphaFunc(u),SigmaAlphaFunc(u))

# %% START ROUTINE CRUDE MC


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

k = 4.0e12 #4 * 10**12 # Fatigue strength normalization factor (?)
m = 3 # Fatigue S-N curve slope (?)

#Delta = stats.lognorm.ppf(np.random.rand(N_MC), s= delta[0], scale = delta[1])

Delta = JDF.LogNormDist(2, np.random.rand(N_MC), delta[0], delta[1])

#X_W = stats.norm.ppf(np.random.rand(N_MC), loc = X_w[0], scale = X_w[1])

X_W = JDF.NormalDist(2, np.random.rand(N_MC), X_w[0], X_w[1])

#X_M = stats.norm.ppf(np.random.rand(N_MC), loc = X_m[0], scale = X_m[1])

X_M = JDF.NormalDist(2, np.random.rand(N_MC), X_m[0], X_m[1])


g_MC = np.zeros(X_W.shape)

N_st = 200  #N short term (100-200)

#iterations = N_st

now = datetime.now()

start_time_str = now.strftime("%H:%M:%S")

start_time = datetime.strptime(start_time_str, "%H:%M:%S")

print("Start of Computation Routine: ", start_time_str)
print("======================================================================")

#Xtrain, Xtest, Ytrain, Ytest, Yscaler = train_surrogate_model(AllInputData, AllTargetData)

#Xscaler, Yscaler = SM.scalers(AllInputData, AllTargetData)

ANNmodel, Xscaler, Yscaler = SM.training(AllInputData, AllTargetData)
 
for i in range(N_MC):
    
    #print("i = ", i)
    U_routine = stats.weibull_min.ppf(np.random.rand(N_st), loc = 0, 
                                      scale = A_weibull*X_W[i], c = k_weibull)
    
    #random.weibull(U, N_st, scale = A_weibull*X_W[i],
    #                           c = k_weibull)
    
    SigmaU_routine = SigmaU(U_routine,np.random.rand(N_st))
    
    Alpha_routine = Alpha(U_routine,np.random.rand(N_st))
    
# =============================================================================
#     routine_df = pd.DataFrame({'U': U_routine, 
#                            'SigmaU': SigmaU_routine,
#                            'Alpha' : Alpha_routine
#                           })
# =============================================================================
    
    routine_array = np.column_stack((U_routine,SigmaU_routine,Alpha_routine))
    
    #put in an array instead of dataframe
    
    #Mx_new = insert routine_df in Surrogate model
    M_x = get_Mx(AllTargetData, routine_array, Xscaler, Yscaler, ANNmodel)
    
    #maybe implement later for improvements
    #M_x[M_x<0] = 0
    
    #don't do in function
    
    #M_x = get_Mx(routine_df, AllInputData, AllTargetData)
    #print("M_x = ", M_x)
    g_MC[i] = Delta[i] - ( 1 / (N_st*k) ) * np.sum( (X_M[i]*M_x)**m )
    
    #print("current g = ", g[i])
    #print("======================================================================")
    #break
print("g_MC = ", g_MC)

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

Nfail = np.sum(g_MC <= 0)

PoF = Nfail/N_MC
beta = JDF.NormalDist(2,PoF)

beta_MC = stats.norm.ppf(1 - PoF)


print("Probability of Failure = ", PoF)
print("Reliability index = ", beta_MC)
print("Number of failure events observed = ", Nfail)

# Plot failure surface based on MC results

# Subplot 1
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(Delta[g_MC > 0], X_M[g_MC > 0], '*b')
axs[0].plot(Delta[g_MC <= 0], X_M[g_MC <= 0], '*r')
axs[0].set_xlabel('$\Delta$')
axs[0].set_ylabel('$X_M$')
#axs[0].set_title('Subplot 1')

# Subplot 2
axs[1].plot(Delta[g_MC > 0], X_W[g_MC > 0], '*b')
axs[1].plot(Delta[g_MC <= 0], X_W[g_MC <= 0], '*r')
axs[1].set_xlabel('$\Delta$')
axs[1].set_ylabel('$X_W$')
#axs[1].set_title('Subplot 2')

# Subplot 3
axs[2].plot(X_W[g_MC > 0], X_M[g_MC > 0], '*b')
axs[2].plot(X_W[g_MC <= 0], X_M[g_MC <= 0], '*r')
axs[2].set_xlabel('$X_W$')
axs[2].set_ylabel('$X_M$')
#axs[2].set_title('Subplot 3')
#plt.suptitle('Crude Monte Carlo', fontsize=16, y=1.02)  # Adjust fontsize and y as needed

plt.tight_layout()
plt.savefig(cur + '\\res\\crude_monte_carlo.eps')
plt.show()


# %% Quasi-MC

# QMC Parameters
qmc_sequence = qmc.Sobol(d=3)  # Sobol sequence with dimension 3
N_QMC = 10**4

sobol_samples = qmc_sequence.random(N_MC)

Delta_Q = JDF.LogNormDist(2, sobol_samples[:, 0], delta[0], delta[1])
X_W_Q = JDF.NormalDist(2, sobol_samples[:, 1], X_w[0], X_w[1])
X_M_Q = JDF.NormalDist(2, sobol_samples[:, 2], X_m[0], X_m[1])


g_QMC = np.zeros(X_W_Q.shape)

#iterations = N_st

now = datetime.now()

start_time_str = now.strftime("%H:%M:%S")

start_time = datetime.strptime(start_time_str, "%H:%M:%S")

print("Start of Quasi Computation Routine: ", start_time_str)
print("======================================================================")

#Xtrain, Xtest, Ytrain, Ytest, Yscaler = train_surrogate_model(AllInputData, AllTargetData)

#Xscaler, Yscaler = SM.scalers(AllInputData, AllTargetData)

#ANNmodel, Xscaler, Yscaler = SM.training(AllInputData, AllTargetData)
 
for i in range(N_QMC):
    
    #print("i = ", i)
    U_routine = stats.weibull_min.ppf(np.random.rand(N_st), loc = 0, 
                                      scale = A_weibull*X_W_Q[i], c = k_weibull)
    
    #random.weibull(U, N_st, scale = A_weibull*X_W[i],
    #                           c = k_weibull)
    
    SigmaU_routine = SigmaU(U_routine,np.random.rand(N_st))
    
    Alpha_routine = Alpha(U_routine,np.random.rand(N_st))
    
# =============================================================================
#     routine_df = pd.DataFrame({'U': U_routine, 
#                            'SigmaU': SigmaU_routine,
#                            'Alpha' : Alpha_routine
#                           })
# =============================================================================
    
    routine_array = np.column_stack((U_routine,SigmaU_routine,Alpha_routine))
    
    #put in an array instead of dataframe
    
    #Mx_new = insert routine_df in Surrogate model
    M_x = get_Mx(AllTargetData, routine_array, Xscaler, Yscaler, ANNmodel)
    
    #maybe implement later for improvements
    #M_x[M_x<0] = 0
    
    #don't do in function
    
    #M_x = get_Mx(routine_df, AllInputData, AllTargetData)
    #print("M_x = ", M_x)
    g_QMC[i] = Delta_Q[i] - ( 1 / (N_st*k) ) * np.sum( (X_M_Q[i]*M_x)**m )
    
    #print("current g = ", g[i])
    #print("======================================================================")
    #break
print("g_QMC = ", g_QMC)

# %% now calculate probability of failure sum(G<= 0)/NMC
# beta = -normaldist(2,P)

now = datetime.now()

end_time_str = now.strftime("%H:%M:%S")

end_time = datetime.strptime(end_time_str, "%H:%M:%S")
    
print("End of Quasi Computation Routine: ", end_time_str)

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

Nfail_Q = np.sum(g_QMC <= 0)

PoF_Q = Nfail_Q/N_QMC
beta_Q = JDF.NormalDist(2,PoF_Q)

beta_QMC = stats.norm.ppf(1 - PoF_Q)


print("Probability of Failure = ", PoF_Q)
print("Reliability index = ", beta_QMC)
print("Number of failure events observed = ", Nfail_Q)

# Plot failure surface based on QMC results


# Subplot 1
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(Delta_Q[g_QMC > 0],X_M_Q[g_QMC >0],'*b')
axs[0].plot(Delta_Q[g_QMC <= 0],X_M_Q[g_QMC <=0],'*r')
axs[0].set_xlabel('$\Delta$')
axs[0].set_ylabel('$X_M$')
#axs[0].set_title('Subplot 1')

# Subplot 2
axs[1].plot(Delta_Q[g_QMC > 0], X_W_Q[g_QMC > 0], '*b')
axs[1].plot(Delta_Q[g_QMC <= 0], X_W_Q[g_QMC <= 0], '*r')
axs[1].set_xlabel('$\Delta$')
axs[1].set_ylabel('$X_W$')
#axs[1].set_title('Subplot 2')

# Subplot 3
axs[2].plot(X_W_Q[g_QMC > 0], X_M_Q[g_QMC > 0], '*b')
axs[2].plot(X_W_Q[g_QMC <= 0], X_M_Q[g_QMC <= 0], '*r')
axs[2].set_xlabel('$X_W$')
axs[2].set_ylabel('$X_M$')
#axs[2].set_title('Subplot 3')

#plt.suptitle('Quasi Monte Carlo', fontsize=16, y=1.02)  # Adjust fontsize and y as needed

plt.tight_layout()
plt.savefig(cur + '\\res\\quasi_monte_carlo.eps')
plt.show()


#Quasi vs Crude
betaMChist = stats.norm.ppf(1 - np.cumsum(g_MC<=0)/np.arange(1,len(g_MC)+1))
betaQMChist = stats.norm.ppf(1 - np.cumsum(g_QMC<=0)/np.arange(1,len(g_QMC)+1))


fig1,axs1 = plt.subplots(1,1,figsize = (6,6))
axs1.plot(betaMChist[:100000], label = 'Crude MC')
axs1.plot(betaQMChist[:100000], label = 'Quasi MC')
axs1.legend()
plt.savefig(cur + '\\res\\crude_vs_quasi_MC.eps')
plt.show()




# %% IMPORTANCE SAMPLING

# Importance Sampling Parameters
shift_factor = 1.03  # Adjust the shift factor based on your problem

g_MC_IS = np.zeros(X_W.shape)

#iterations = N_st

now = datetime.now()

start_time_str = now.strftime("%H:%M:%S")

start_time = datetime.strptime(start_time_str, "%H:%M:%S")

print("Start of Importance Sampling Crude Computation Routine: ", start_time_str)
print("======================================================================")


for i in range(N_MC):
    # Shift the sample closer to the limit state
    U_routine = stats.weibull_min.ppf(np.random.rand(N_st), loc=0, scale=A_weibull * X_W[i], c=k_weibull)

    # Apply the shift factor for importance sampling
    U_routine_shifted = U_routine - shift_factor * (U_routine - Delta[i])

    SigmaU_routine = SigmaU(U_routine_shifted, np.random.rand(N_st))
    Alpha_routine = Alpha(U_routine_shifted, np.random.rand(N_st))

    routine_array = np.column_stack((U_routine_shifted, SigmaU_routine, Alpha_routine))

    M_x = get_Mx(AllTargetData, routine_array, Xscaler, Yscaler, ANNmodel)

    g_MC_IS[i] = Delta[i] - (1 / (N_st * k)) * np.sum((X_M[i] * M_x)**m)


print("g_MC_IS = ", g_MC_IS)

# now calculate probability of failure sum(G<= 0)/NMC
# beta = -normaldist(2,P)

now = datetime.now()

end_time_str = now.strftime("%H:%M:%S")

end_time = datetime.strptime(end_time_str, "%H:%M:%S")
    
print("End of Crude IS Computation Routine: ", end_time_str)

print("+++++++++++++++++++++++++++++++++++++++++++++")

Nfail_IS_C = np.sum(g_MC_IS <= 0)

PoF_IS_C = Nfail_IS_C/N_MC

beta_IS_C = stats.norm.ppf(1 - PoF_IS_C)


print("Probability of Failure = ", PoF_IS_C)
print("Reliability index = ", beta_IS_C)
print("Number of failure events observed = ", Nfail_IS_C)

# Plot failure surface based on QMC results


# Subplot 1
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(Delta[g_MC_IS > 0],X_M[g_MC_IS >0],'*b')
axs[0].plot(Delta[g_MC_IS <= 0],X_M[g_MC_IS <=0],'*r')
axs[0].set_xlabel('$\Delta$')
axs[0].set_ylabel('$X_M$')
#axs[0].set_title('Subplot 1')

# Subplot 2
axs[1].plot(Delta[g_MC_IS > 0], X_W[g_MC_IS > 0], '*b')
axs[1].plot(Delta[g_MC_IS <= 0], X_W[g_MC_IS <= 0], '*r')
axs[1].set_xlabel('$\Delta$')
axs[1].set_ylabel('$X_W$')
#axs[1].set_title('Subplot 2')

# Subplot 3
axs[2].plot(X_W[g_MC_IS > 0], X_M[g_MC_IS > 0], '*b')
axs[2].plot(X_W[g_MC_IS <= 0], X_M[g_MC_IS <= 0], '*r')
axs[2].set_xlabel('$X_W$')
axs[2].set_ylabel('$X_M$')
#axs[2].set_title('Subplot 3')

#plt.suptitle('Importance Sampling: Crude Monte Carlo', fontsize=16, y=1.02)  # Adjust fontsize and y as needed

plt.tight_layout()
plt.savefig(cur + '\\res\\imp_sampling_crude_monte_carlo.eps')
plt.show()


# Importance Sampling Parameters
shift_factor = 1.03  # Adjust the shift factor based on your problem

g_QMC_IS = np.zeros(X_W_Q.shape)

#iterations = N_st

now = datetime.now()

start_time_str = now.strftime("%H:%M:%S")

start_time = datetime.strptime(start_time_str, "%H:%M:%S")

print("Start of Importance Sampling Quasi Computation Routine: ", start_time_str)
print("======================================================================")


for i in range(N_QMC):
    # Shift the sample closer to the limit state
    U_routine = stats.weibull_min.ppf(np.random.rand(N_st), loc=0, scale=A_weibull * X_W_Q[i], c=k_weibull)

    # Apply the shift factor for importance sampling
    U_routine_shifted = U_routine - shift_factor * (U_routine - Delta_Q[i])

    SigmaU_routine = SigmaU(U_routine_shifted, np.random.rand(N_st))
    Alpha_routine = Alpha(U_routine_shifted, np.random.rand(N_st))

    routine_array = np.column_stack((U_routine_shifted, SigmaU_routine, Alpha_routine))

    M_x = get_Mx(AllTargetData, routine_array, Xscaler, Yscaler, ANNmodel)

    g_QMC_IS[i] = Delta_Q[i] - (1 / (N_st * k)) * np.sum((X_M_Q[i] * M_x)**m)


print("g_QMC_IS = ", g_QMC_IS)

# now calculate probability of failure sum(G<= 0)/NMC
# beta = -normaldist(2,P)

now = datetime.now()

end_time_str = now.strftime("%H:%M:%S")

end_time = datetime.strptime(end_time_str, "%H:%M:%S")
    
print("End of Quasi IS Computation Routine: ", end_time_str)

print("+++++++++++++++++++++++++++++++++++++++++++++")

Nfail_IS_Q = np.sum(g_QMC_IS <= 0)

PoF_IS_Q = Nfail_IS_Q/N_QMC

beta_IS_Q = stats.norm.ppf(1 - PoF_IS_Q)


print("Probability of Failure = ", PoF_IS_Q)
print("Reliability index = ", beta_IS_Q)
print("Number of failure events observed = ", Nfail_IS_Q)

# Plot failure surface based on QMC results


# Subplot 1
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(Delta_Q[g_QMC_IS > 0],X_M_Q[g_QMC_IS >0],'*b')
axs[0].plot(Delta_Q[g_QMC_IS <= 0],X_M_Q[g_QMC_IS <=0],'*r')
axs[0].set_xlabel('$\Delta$')
axs[0].set_ylabel('$X_M$')
#axs[0].set_title('Subplot 1')

# Subplot 2
axs[1].plot(Delta_Q[g_QMC_IS > 0], X_W_Q[g_QMC_IS > 0], '*b')
axs[1].plot(Delta_Q[g_QMC_IS <= 0], X_W_Q[g_QMC_IS <= 0], '*r')
axs[1].set_xlabel('$\Delta$')
axs[1].set_ylabel('$X_W$')
#axs[1].set_title('Subplot 2')

# Subplot 3
axs[2].plot(X_W_Q[g_QMC_IS > 0], X_M_Q[g_QMC_IS > 0], '*b')
axs[2].plot(X_W_Q[g_QMC_IS <= 0], X_M_Q[g_QMC_IS <= 0], '*r')
axs[2].set_xlabel('$X_W$')
axs[2].set_ylabel('$X_M$')
#axs[2].set_title('Subplot 3')

#plt.suptitle('Importance Sampling: Quasi Monte Carlo', fontsize=16, y=1.02)  # Adjust fontsize and y as needed

plt.tight_layout()
plt.savefig(cur + '\\res\\imp_sampling_quasi_monte_carlo.eps')
plt.show()



