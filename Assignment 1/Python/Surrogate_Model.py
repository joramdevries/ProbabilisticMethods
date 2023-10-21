# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:36:26 2023

@author: joram
"""
#%% IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network
import os


cur = os.getcwd()

#%% IMPORTING DATA

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
print('Feature names: ', FeatureNames)
print('Dependent variable names: ', DependentVariableNames)
print(AllInputData.iloc[:,0].values)

# %% PLOTTING U

fig2 = plt.figure(2, figsize = (18,9))

for i in range(DependentVariableNames.shape[0]):
    axi = fig2.add_subplot(2,4,i+1)
    plt.title(DependentVariableNames[i])
    plt.plot(AllInputData.U,AllTargetData.iloc[:,i],'y.',markersize = 3)
    plt.xlabel('Mean wind speed [m/s]')
    #plt.ylabel(DependentVariableNames[i])
plt.tight_layout()
plt.savefig(cur + '\\res\\Surrogate_model_U.eps')
plt.show()


# %% PLOTTING SIGMA

fig3 = plt.figure(3, figsize = (18,9))

for i in range(DependentVariableNames.shape[0]):
    axi = fig3.add_subplot(2,4,i+1)
    plt.title(DependentVariableNames[i])
    plt.plot(AllInputData.SigmaU,AllTargetData.iloc[:,i],'g.',markersize = 3)
    plt.xlabel('Wind standard deviation [m/s]')
    #plt.ylabel(DependentVariableNames[i])
plt.tight_layout()
plt.savefig(cur + '\\res\\Surrogate_model_sigma.eps')
plt.show()


# %% PLOTTING SHEAR

fig4 = plt.figure(3, figsize = (18,9))

for i in range(DependentVariableNames.shape[0]):
    axi = fig4.add_subplot(2,4,i+1)
    plt.title(DependentVariableNames[i])
    plt.plot(AllInputData.Alpha,AllTargetData.iloc[:,i],'r.',markersize = 3)
    plt.xlabel('Wind shear exponent [-]')
    #plt.ylabel(DependentVariableNames[i])
plt.tight_layout()
plt.savefig(cur + '\\res\\Surrogate_model_shear.eps')
plt.show()


# %% POLYNOMIAL MODEL



# Building a design matrix for a polynomial of 3rd order
def DesignMatrixO3(X):
    ndim = X.shape[1] 
    npoints = X.shape[0]
    m = int(((ndim-1)/2)*ndim)
    Xmatrix = np.zeros((npoints,3*ndim + 3*m + 1))
    columncount = 0
    Xmatrix[:,columncount] = np.ones(npoints)
    for i in range(ndim):
        columncount+=1
        Xmatrix[:,columncount] = X[:,i]

    for i in range(ndim -1):
        for j in range(i+1,ndim):
            columncount+= 1
            Xmatrix[:,columncount] = X[:,i]*X[:,j]

    for i in range(ndim):
        columncount+= 1
        Xmatrix[:,columncount] = X[:,i]**2

    for i in range(ndim-1):
        for j in range(i+1,ndim):
            columncount+= 1
            Xmatrix[:,columncount] = (X[:,i]**2)*X[:,j]

    for i in range(ndim-1):
        for j in range(i+1,ndim):
            columncount+= 1
            Xmatrix[:,columncount] = X[:,i]*(X[:,j]**2)

    for i in range(ndim):
        columncount+=1
        Xmatrix[:,columncount] = X[:,i]**3
    return Xmatrix


def PredictPolyO3(X,Alsq):
    Xmatrix = DesignMatrixO3(X)
    Y = np.dot(Xmatrix,Alsq)
    return Y

# %% PLOTTING U

# MAKE A 3-RD ORDER POLYNOMIAL FIT TO THE DATA USING THE HELPER FUNCTIONS GIVEN ABOVE


## WIND SPEED U
for i in range(DependentVariableNames.shape[0]):
    Y1 = AllTargetData[DependentVariableNames[i]]

    # BEGIN CODE HERE
    Xmatrix = DesignMatrixO3(AllInputData.values)

    XX = np.dot(Xmatrix.T,Xmatrix)
    XY = np.dot(Xmatrix.T,Y1)
    Alsq = np.linalg.lstsq(XX,XY, rcond = None)
    Alsq = Alsq[0]
    Alsq.shape

    Ypred_O3 = PredictPolyO3(AllInputData.values, Alsq)
    # END CODE HERE

    plt.rc('font', size=14) 
    fig2a,axs2a = plt.subplots(1,2,figsize = (16,8))
    plt.setp(axs2a[0], title = 'Dependence vs. wind speed', xlabel = 'Mean wind speed [m/s]',ylabel = f'{DependentVariableNames[i]}')
    plt.setp(axs2a[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
    axs2a[0].plot(AllInputData['U'],AllTargetData[DependentVariableNames[i]],'o',markersize = 4,color = 'y')
    axs2a[0].plot(AllInputData['U'],Ypred_O3,'x',markersize = 4,color = 'purple')
    axs2a[0].legend(['Input data','Model predictions'])
    axs2a[1].plot(AllTargetData[DependentVariableNames[i]],Ypred_O3,'ok',markersize = 4)
    axs2a[1].plot(np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),\
                 np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),'-y',linewidth = 4)
    axs2a[1].legend(['Point-to-point comparisons','1:1 relation'])
    plt.tight_layout()  
    plt.savefig(cur + '\\res\\Surrogate_model_U_2.eps')           
    plt.show()
    
    
#%% PLOTTING SIGMA

# MAKE A 3-RD ORDER POLYNOMIAL FIT TO THE DATA USING THE HELPER FUNCTIONS GIVEN ABOVE


## WIND STANDARD DEVIATION SIGMA
for i in range(DependentVariableNames.shape[0]):
    Y1 = AllTargetData[DependentVariableNames[i]]

    # BEGIN CODE HERE
    Xmatrix = DesignMatrixO3(AllInputData.values)

    XX = np.dot(Xmatrix.T,Xmatrix)
    XY = np.dot(Xmatrix.T,Y1)
    Alsq = np.linalg.lstsq(XX,XY, rcond = None)
    Alsq = Alsq[0]
    Alsq.shape

    Ypred_O3 = PredictPolyO3(AllInputData.values, Alsq)
    # END CODE HERE

    plt.rc('font', size=14) 
    fig2a,axs2a = plt.subplots(1,2,figsize = (16,8))
    plt.setp(axs2a[0], title = 'Dependence vs. standard deviation', xlabel = 'Standard Deviation [m/s]',ylabel = f'{DependentVariableNames[i]}')
    plt.setp(axs2a[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
    axs2a[0].plot(AllInputData['SigmaU'],AllTargetData[DependentVariableNames[i]],'o',markersize = 4,color = 'g')
    axs2a[0].plot(AllInputData['SigmaU'],Ypred_O3,'x',markersize = 4,color = 'orange')
    axs2a[0].legend(['Input data','Model predictions'])
    axs2a[1].plot(AllTargetData[DependentVariableNames[i]],Ypred_O3,'ok',markersize = 4)
    axs2a[1].plot(np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),\
                 np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),'-g',linewidth = 4)
    axs2a[1].legend(['Point-to-point comparisons','1:1 relation'])
    plt.tight_layout()  
    plt.savefig(cur + '\\res\\Surrogate_model_sigma_2.eps')           
    plt.show()
    
#%% PLOTTING SHEAR

# MAKE A 3-RD ORDER POLYNOMIAL FIT TO THE DATA USING THE HELPER FUNCTIONS GIVEN ABOVE


## WIND SHEAR EXPONENT ALPHA
for i in range(DependentVariableNames.shape[0]):
    Y1 = AllTargetData[DependentVariableNames[i]]

    # BEGIN CODE HERE
    Xmatrix = DesignMatrixO3(AllInputData.values)

    XX = np.dot(Xmatrix.T,Xmatrix)
    XY = np.dot(Xmatrix.T,Y1)
    Alsq = np.linalg.lstsq(XX,XY, rcond = None)
    Alsq = Alsq[0]
    Alsq.shape

    Ypred_O3 = PredictPolyO3(AllInputData.values, Alsq)
    # END CODE HERE

    plt.rc('font', size=14) 
    fig2a,axs2a = plt.subplots(1,2,figsize = (16,8))
    plt.setp(axs2a[0], title = 'Dependence vs. wind shear exponent', xlabel = 'Wind shear exponent [-]',ylabel = f'{DependentVariableNames[i]}')
    plt.setp(axs2a[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
    axs2a[0].plot(AllInputData['Alpha'],AllTargetData[DependentVariableNames[i]],'o',markersize = 4,color = 'red')
    axs2a[0].plot(AllInputData['Alpha'],Ypred_O3,'*',markersize = 4,color = 'blue')
    axs2a[0].legend(['Input data','Model predictions'])
    axs2a[1].plot(AllTargetData[DependentVariableNames[i]],Ypred_O3,'ok',markersize = 4)
    axs2a[1].plot(np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),\
                 np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),'-r', linewidth = 4)
    axs2a[1].legend(['Point-to-point comparisons','1:1 relation'])
    plt.tight_layout() 
    plt.savefig(cur + '\\res\\Surrogate_model_shear_2.eps')            
    plt.show()