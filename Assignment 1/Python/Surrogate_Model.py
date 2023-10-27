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
import sklearn.neural_network as nn
import os


# %% MAIN LOOP
if __name__ == "__main__":
    # %% SELECTION MENU
    
    plots = True
    training_model = True
    
    
    # %% CUR
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
    
    
    # %% Select if you want plots
    
    if plots:
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
            plt.xlabel('Turbulence [m/s]')
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
        
        
        # %% PLOTTING Mx
        
        FeatureNames = FeatureNames[:3]
        
        fig4 = plt.figure(3, figsize = (18,9))
        j=0
        for i in FeatureNames:
            axi = fig4.add_subplot(1,3,j+1)
        
            #plt.title(DependentVariableNames[i])
            if i == 'U':
                plt.plot(AllInputData.U,AllTargetData.iloc[:,0],'y.',markersize = 3)
                plt.xlabel('Mean wind speed [m/s]')
                plt.ylabel('Blade root flapwise moment $M_x$ [Nm]')
            if i == 'SigmaU':
                plt.plot(AllInputData.SigmaU,AllTargetData.iloc[:,0],'g.',markersize = 3)
                plt.xlabel('Turbulence [m/s]')
            if i == 'Alpha':
                plt.plot(AllInputData.Alpha,AllTargetData.iloc[:,0],'r.',markersize = 3)
                plt.xlabel('Wind shear exponent [-]')
            j += 1
        
            #plt.ylabel(DependentVariableNames[i])
        
        plt.tight_layout()
        plt.savefig(cur + '\\res\\Surrogate_model_Mx.eps')
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
            
            if DependentVariableNames[i] == 'Blade_root_flapwise_M_x':
                y_label_for_u = 'Blade root flapwise moment $M_x$ [Nm]'
            else:
                y_label_for_u = f'{DependentVariableNames[i]}'
        
            plt.rc('font', size=14) 
            fig2a,axs2a = plt.subplots(1,2,figsize = (16,8))
            plt.setp(axs2a[0], title = 'Dependence vs. wind speed', xlabel = 'Mean wind speed [m/s]',ylabel = y_label_for_u)
            plt.setp(axs2a[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
            axs2a[0].plot(AllInputData['U'],AllTargetData[DependentVariableNames[i]],'o',markersize = 4,color = 'y')
            axs2a[0].plot(AllInputData['U'],Ypred_O3,'x',markersize = 4,color = 'purple')
            axs2a[0].legend(['Input data','Model predictions'])
            axs2a[1].plot(AllTargetData[DependentVariableNames[i]],Ypred_O3,'ok',markersize = 4)
            axs2a[1].plot(np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),\
                         np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),'-y',linewidth = 4)
            axs2a[1].legend(['Point-to-point comparisons','1:1 relation'])
            plt.tight_layout()  
            plt.savefig(cur + f'\\res\\Surrogate_model_U_2_{i}.eps')           
            plt.show()
            
            
        #%% PLOTTING SIGMA
        
        # MAKE A 3-RD ORDER POLYNOMIAL FIT TO THE DATA USING THE HELPER FUNCTIONS GIVEN ABOVE
        
        
        ## Turbulence SIGMA
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
            plt.setp(axs2a[0], title = 'Dependence vs. Turbulence', xlabel = 'Turbulence [m/s]',ylabel = f'{DependentVariableNames[i]}')
            plt.setp(axs2a[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
            axs2a[0].plot(AllInputData['SigmaU'],AllTargetData[DependentVariableNames[i]],'o',markersize = 4,color = 'g')
            axs2a[0].plot(AllInputData['SigmaU'],Ypred_O3,'x',markersize = 4,color = 'orange')
            axs2a[0].legend(['Input data','Model predictions'])
            axs2a[1].plot(AllTargetData[DependentVariableNames[i]],Ypred_O3,'ok',markersize = 4)
            axs2a[1].plot(np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),\
                         np.array([np.min(AllTargetData[DependentVariableNames[i]]), np.max(AllTargetData[DependentVariableNames[i]])]),'-g',linewidth = 4)
            axs2a[1].legend(['Point-to-point comparisons','1:1 relation'])
            plt.tight_layout()  
            plt.savefig(cur + f'\\res\\Surrogate_model_sigma_2_{i}.eps')           
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
            plt.savefig(cur + f'\\res\\Surrogate_model_shear_2_{i}.eps')            
            plt.show()
        
      
    # %% Select if you want to train the model
    
    if training_model:
    
        # %% NEUTAL NETWORK MODEL
        
        # SKLEARN Neural Network MLP regressor model
        
        Y1 = AllTargetData['Blade_root_flapwise_M_x']
        
        ANNmodel = nn.MLPRegressor()
        
        ANNmodel.get_params()
        
        Xscaler = sklearn.preprocessing.StandardScaler()
        Yscaler = sklearn.preprocessing.StandardScaler()
        Xscaler = Xscaler.fit(AllInputData)
        Yscaler = Yscaler.fit(AllTargetData['Blade_root_flapwise_M_x'].values.reshape(-1, 1))
        
        TrainTestRatio = 0.8
        N = len(AllInputData)
        Xtrain = Xscaler.transform(AllInputData.values[:int(N*TrainTestRatio),:])
        Xtest = Xscaler.transform(AllInputData.values[int(N*TrainTestRatio):,:])
        
        Ytrain = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[:int(N*TrainTestRatio)].reshape(-1,1))
        Ytest = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[int(N*TrainTestRatio):].reshape(-1,1))
        
        
        ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)
        # BEGIN CODE HERE
        ANNmodel.fit(Xtrain, Ytrain.ravel())
        
        print( 'Train set r-square: ' + str(ANNmodel.score(Xtrain,Ytrain)))
        print( 'Test set r-square: ' + str(ANNmodel.score(Xtest,Ytest)))
        
        Yout = Yscaler.inverse_transform(ANNmodel.predict(Xtrain).reshape(-1, 1))
        Yout_test = Yscaler.inverse_transform(ANNmodel.predict(Xtest).reshape(-1, 1))
        
        plt.hist(Yout,50)
        plt.show()
        
        plt.rc('font', size=14) 
        fig3,axs3 = plt.subplots(1,2,figsize = (16,8))
        plt.setp(axs3[0], title = 'Dependence vs. wind speed', xlabel = 'Mean wind speed [m/s]',ylabel = 'Blade root flapwise moment $M_x$ [Nm]')
        plt.setp(axs3[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
        axs3[0].plot(Xtest[:,0],Yscaler.inverse_transform(Ytest),'o',markersize = 4,color = 'y')
        axs3[0].plot(Xtest[:,0],Yout_test,'*',markersize = 4,color = 'purple')
        axs3[0].legend(['Input data','Model predictions'])
        axs3[1].plot(Yscaler.inverse_transform(Ytest),Yout_test,'ok',markersize = 4)
        axs3[1].plot(np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),\
                     np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),'-y', linewidth =3)
        axs3[1].legend(['Point-to-point comparisons','1:1 relation'])
        plt.tight_layout()
        plt.savefig(cur + '\\res\\training_1.eps')              
        plt.show()
        
        plt.rc('font', size=14) 
        fig4,axs4 = plt.subplots(1,2,figsize = (16,8))
        plt.setp(axs4[0], title = 'Dependence vs. Turbulence', xlabel = 'Turbulence [m/s]',ylabel = 'Blade root flapwise moment $M_x$ [Nm]')
        plt.setp(axs4[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
        axs4[0].plot(Xtest[:,1],Yscaler.inverse_transform(Ytest),'o',markersize = 4,color = 'g')
        axs4[0].plot(Xtest[:,1],Yout_test,'*',markersize = 4,color = 'orange')
        axs4[0].legend(['Input data','Model predictions'])
        axs4[1].plot(Yscaler.inverse_transform(Ytest),Yout_test,'ok',markersize = 4)
        axs4[1].plot(np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),\
                     np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),'-g', linewidth =3)
        axs4[1].legend(['Point-to-point comparisons','1:1 relation'])
        plt.tight_layout() 
        plt.savefig(cur + '\\res\\training_2.eps')             
        plt.show()
        
        plt.rc('font', size=14) 
        fig5,axs5 = plt.subplots(1,2,figsize = (16,8))
        plt.setp(axs5[0], title = 'Dependence vs. wind shear', xlabel = 'Wind shear exponent [-]',ylabel = 'Blade root flapwise moment $M_x$ [Nm]')
        plt.setp(axs5[1], title = 'Correlation (y-y) plot', xlabel = 'Input data',ylabel = 'Model predictions')
        axs5[0].plot(Xtest[:,2],Yscaler.inverse_transform(Ytest),'o',markersize = 4,color = 'r')
        axs5[0].plot(Xtest[:,2],Yout_test,'*',markersize = 4,color = 'blue')
        axs5[0].legend(['Input data','Model predictions'])
        axs5[1].plot(Yscaler.inverse_transform(Ytest),Yout_test,'ok',markersize = 4)
        axs5[1].plot(np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),\
                     np.array([np.min(AllTargetData['Blade_root_flapwise_M_x']), np.max(AllTargetData['Blade_root_flapwise_M_x'])]),'-r', linewidth =3)
        axs5[1].legend(['Point-to-point comparisons','1:1 relation'])
        plt.tight_layout()
        plt.savefig(cur + '\\res\\training_3.eps')              
        plt.show()
    
def scalers(AllInputData, AllTargetData):
    
    ANNmodel = nn.MLPRegressor()

    ANNmodel.get_params()
    
    Xscaler = sklearn.preprocessing.StandardScaler()
    Yscaler = sklearn.preprocessing.StandardScaler()
    Xscaler = Xscaler.fit(AllInputData)
    Yscaler = Yscaler.fit(AllTargetData['Blade_root_flapwise_M_x'].values.reshape(-1, 1))
    
    return Xscaler, Yscaler

def training(AllInputData, AllTargetData):
    
    ANNmodel = nn.MLPRegressor()

    ANNmodel.get_params()
    
    Xscaler = sklearn.preprocessing.StandardScaler()
    Yscaler = sklearn.preprocessing.StandardScaler()
    Xscaler = Xscaler.fit(AllInputData)
    Yscaler = Yscaler.fit(AllTargetData['Blade_root_flapwise_M_x'].values.reshape(-1, 1))
    
    TrainTestRatio = 0.8
    N = len(AllInputData)
    Xtrain = Xscaler.transform(AllInputData.values[:int(N*TrainTestRatio),:])
    Xtest = Xscaler.transform(AllInputData.values[int(N*TrainTestRatio):,:])
    
    Ytrain = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[:int(N*TrainTestRatio)].reshape(-1,1))
    Ytest = Yscaler.transform(AllTargetData['Blade_root_flapwise_M_x'].values[int(N*TrainTestRatio):].reshape(-1,1))
    
    
    ANNmodel.set_params(learning_rate_init = 0.01, activation = 'relu',tol = 1e-6,n_iter_no_change = 10, hidden_layer_sizes = (12,12), validation_fraction = 0.1)
    # BEGIN CODE HERE
    ANNmodel.fit(Xtrain, Ytrain.ravel())
    
    print( 'Train set r-square: ' + str(ANNmodel.score(Xtrain,Ytrain)))
    print( 'Test set r-square: ' + str(ANNmodel.score(Xtest,Ytest)))
    
    Yout = Yscaler.inverse_transform(ANNmodel.predict(Xtrain).reshape(-1, 1))
    Yout_test = Yscaler.inverse_transform(ANNmodel.predict(Xtest).reshape(-1, 1))
    
    return ANNmodel, Xscaler, Yscaler