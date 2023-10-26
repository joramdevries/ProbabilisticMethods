# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:14:45 2023

@author: joram
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network
import seaborn as sns
import os

# %% DEFINITION

def Weibull_parameters(WindData):
    
    # WIND SPEED DISTRIBUTION FIT
    Wsp0 = np.asarray(WindData['Wsp'])
    print("Wsp0 = ", Wsp0)
    print("---------------------------------------")
    
    #Wsp_mean = np.mean(Wsp0)
    #Wsp_std = np.std(Wsp0)
    
    WeibLikelihoodFunc = lambda theta: - np.sum( np.log( stats.weibull_min.pdf(Wsp0, loc = 0, scale = theta[0], c = theta[1])) )
    print("WeibLikelihoodFunc = ", WeibLikelihoodFunc)
    Weib0 = scipy.optimize.minimize(WeibLikelihoodFunc, [5,1])
    print("---------------------------------------")
    
    print("Weib0 = ", Weib0)
    WeibullA = Weib0.x[0]
    Weibullk = Weib0.x[1]
    print("---------------------------------------")
    print("WeibullA = ", WeibullA)
    print("Weibullk = ", Weibullk)
    
    return WeibullA, Weibullk

# Helper function - Normal distribution -> first variable is the mean, next one is the std_dev
def NormalDist(task,x,mu=0,sigma=1):
    import numpy as np
    if task == 0: # PDF
        y = (1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x - mu)**2)/(2.0*(sigma**2)))
    elif task == 1: # Cumulative
        from scipy.special import erf
        y = 0.5*(1.0 + erf((x - mu)/(sigma*np.sqrt(2))))
    elif task == 2: # Inverse
        from scipy.special import erfinv
        y = mu + sigma*np.sqrt(2)*erfinv(2*x - 1)        
    return y

# JUST A Helper function - lognormal distribution -> this implementation makes it a function of the variable mean and the std_dev
def LogNormDist(task,x,mu,sigma):
    import numpy as np
    tol = 1e-16
    mu = np.asarray(mu)
    mu[mu<tol] = tol
    Eps   = np.sqrt(np.log( 1.0+(sigma/mu)**2 ) ) #-> getting the true functions
    Ksi   = np.log(mu)-0.5*Eps**2
    if task == 0: # PDF
        x[x<=0] = 1e-8
        u =(np.log(x)-Ksi)/Eps
        y = np.exp(-u*u/2.0)/(Eps*x*np.sqrt(2.0*np.pi))
    elif task == 1: # Cummulative
        x[x<=0] = 1e-8
        u =(np.log(x)-Ksi)/Eps
        y= NormalDist(1, u)
    elif task == 2: # Inverse
        y= np.exp(Ksi+Eps*NormalDist(2, x))
    
    return y

# %% main
if __name__ == "__main__":
    cur = os.getcwd()
    
    #%% PLOT SELECTION
    
    mean_sigma_plot = False
    binned_plot = True
    turbulence_plot = True
    monte_carlo_plot = True
    
    
    #%% IMPORT DATA & DROP NAN
    
    WindData = pd.read_csv('HovsoreData_Sonic_100m_2004-2013.csv')
    
    # Filter rows where 'Wsp' is less than or equal to 35 m/s
    filtered_WindData = WindData[WindData['Wsp'] <= 35]
    
    filtered_WindData.loc[filtered_WindData["TI"] <= 0.001] = np.nan
    
    filtered_WindData = filtered_WindData.dropna()
    
    # Save the filtered data to a new CSV file
    filtered_csv_file = 'FilteredWindData.csv'
    filtered_WindData.to_csv(filtered_csv_file, index=False)
    
    WindData = filtered_WindData
    
    WindData["SigmaU"] = WindData['Wsp']*WindData['TI']
    
    # %% FUNCTIONS
    
    # Helper function - Normal distribution -> first variable is the mean, next one is the std_dev
    def NormalDist(task,x,mu=0,sigma=1):
        import numpy as np
        if task == 0: # PDF
            y = (1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x - mu)**2)/(2.0*(sigma**2)))
        elif task == 1: # Cumulative
            from scipy.special import erf
            y = 0.5*(1.0 + erf((x - mu)/(sigma*np.sqrt(2))))
        elif task == 2: # Inverse
            from scipy.special import erfinv
            y = mu + sigma*np.sqrt(2)*erfinv(2*x - 1)        
        return y
    
    # JUST A Helper function - lognormal distribution -> this implementation makes it a function of the variable mean and the std_dev
    def LogNormDist(task,x,mu,sigma):
        import numpy as np
        tol = 1e-16
        mu = np.asarray(mu)
        mu[mu<tol] = tol
        Eps   = np.sqrt(np.log( 1.0+(sigma/mu)**2 ) ) #-> getting the true functions
        Ksi   = np.log(mu)-0.5*Eps**2
        if task == 0: # PDF
            x[x<=0] = 1e-8
            u =(np.log(x)-Ksi)/Eps
            y = np.exp(-u*u/2.0)/(Eps*x*np.sqrt(2.0*np.pi))
        elif task == 1: # Cummulative
            x[x<=0] = 1e-8
            u =(np.log(x)-Ksi)/Eps
            y= NormalDist(1, u)
        elif task == 2: # Inverse
            y= np.exp(Ksi+Eps*NormalDist(2, x))
        
        return y
    
    def likelihood(mean, std_dev, data):
        likelihood_values = 1.0 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((data - mean) / std_dev) ** 2)
        return np.prod(likelihood_values)
    
    #%% PLOTTING MEAN & SIGMA
    
    if mean_sigma_plot:
        mean_winddata = np.mean(WindData['Wsp'])
        print(mean_winddata)
        print("--------------------------------")
        print(WindData['Wsp'])
        #WindData["SigmaU"] = WindData['Wsp']*WindData['TI']
        print("--------------------------------")
        
        print(WindData["SigmaU"])
        print("--------------------------------")
        print(WindData)
        
        # Scatter plot of Wsp and SigmaU
        plt.scatter(WindData['Wsp'], WindData["SigmaU"])
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel(r'$\sigma_U$', fontsize=14)
        plt.title('Scatter plot of Wsp vs. SigmaU')
        plt.savefig(cur + '\\res\\Scatter_plot_Wsp_vs_SigmaU.eps')
        plt.show()
        
        # # Scatter plot of Wsp and SigmaU -- HEX
        # # sns.set(style="whitegrid", palette="pastel")
        # sns.jointplot(x=WindData['Wsp'], y=WindData['SigmaU'], kind='hex', color='lightgreen')#.plot_joint(sns.kdeplot)
        # plt.xlabel('Wind Speed [m/s]', fontsize=14)
        # plt.ylabel(r'$\sigma_U$', fontsize=14)
        # # plt.suptitle('Joint Plot of Wsp vs. SigmaU', fontsize=16, y=0.85)
        # plt.savefig(cur + '\\res\\Joint_plot_Wsp_vs_SigmaU.eps')
        # plt.show()
        
        # Scatter plot of Wsp and SigmaU -- Scatter + KDE
        # sns.set(style="whitegrid", palette="pastel")
        sns.jointplot(x=WindData['Wsp'], y=WindData['SigmaU'], kind='scatter', color='red').plot_joint(sns.kdeplot)
        plt.xlabel('Wind Speed [m/s]', fontsize=14)
        plt.ylabel(r'$\sigma_U$', fontsize=14)
        # plt.suptitle('Joint Plot of Wsp vs. SigmaU', fontsize=16, y=0.80)
        plt.savefig(cur + '\\res\\Joint_plot_2_Wsp_vs_SigmaU.eps')
        plt.show()
        
        
    # %% PLOTTING BINNED
    
    if binned_plot:    
        # WIND SPEED DISTRIBUTION FIT
        Wsp0 = np.asarray(WindData['Wsp'])
        print("Wsp0 = ", Wsp0)
        print("---------------------------------------")
        
        #Wsp_mean = np.mean(Wsp0)
        #Wsp_std = np.std(Wsp0)
        
        WeibLikelihoodFunc = lambda theta: - np.sum( np.log( stats.weibull_min.pdf(Wsp0, loc = 0, scale = theta[0], c = theta[1])) )
        print("WeibLikelihoodFunc = ", WeibLikelihoodFunc)
        Weib0 = scipy.optimize.minimize(WeibLikelihoodFunc, [5,1])
        print("---------------------------------------")
        
        print("Weib0 = ", Weib0)
        WeibullA = Weib0.x[0]
        Weibullk = Weib0.x[1]
        print("---------------------------------------")
        print("WeibullA = ", WeibullA)
        print("Weibullk = ", Weibullk)
        
        WspBinEdges = np.arange(3.5,33.5,1)
        WspBinCenters = WspBinEdges[:-1] + 0.5
        
        MuSigmaBinned = np.zeros(len(WspBinCenters))
        SigmaSigmaBinned = np.zeros(len(WspBinCenters))
        
        nData = len(WindData['Wsp'])
            
        # Per wind speed
        for iWsp in range(len(WspBinCenters)): #we are simply selecting here (we want higher than the lower edge, and lower than upper edge)
            WspBinSelection = (WindData['Wsp'] > WspBinEdges[iWsp]) & (WindData['Wsp'] <= WspBinEdges[iWsp + 1]) #select data that falls in the bin
            MuSigmaBinned[iWsp] = np.mean(WindData.loc[WspBinSelection,'SigmaU'])
            SigmaSigmaBinned[iWsp] = np.std(WindData.loc[WspBinSelection,'SigmaU'])
            
        plt.plot(WspBinCenters, MuSigmaBinned, "-ok")
        plt.plot(WspBinCenters, SigmaSigmaBinned, "-xb") # for turbulence this is a common assumption
        plt.savefig(cur + '\\res\\binned_plot_Wsp_vs_Sigma.eps')
    
        Mudatax = WspBinCenters[~np.isnan(MuSigmaBinned)]
        Mudatay = MuSigmaBinned[~np.isnan(MuSigmaBinned)]
        
        pMu = np.polyfit(Mudatax,Mudatay,2)
        
        # lets plot the prediction in red
        plt.plot(WspBinCenters, MuSigmaBinned, "-ok")
        plt.plot(WspBinCenters, SigmaSigmaBinned, "-xb") # for turbulence this is a common assumption
        plt.plot(WspBinCenters, (pMu[0]*WspBinCenters**2 + pMu[1]*WspBinCenters + pMu[2]), '-r')
        plt.savefig(cur + '\\res\\bin_center_plot.eps')
        
        SigmaSigmaRef = np.mean(SigmaSigmaBinned)
        
        # plotting sigma
        plt.plot(WspBinCenters, MuSigmaBinned, "-ok")
        plt.plot(WspBinCenters, SigmaSigmaBinned, "-xb") # for turbulence this is a common assumption
        plt.plot(WspBinCenters, (pMu[0]*WspBinCenters**2 + pMu[1]*WspBinCenters + pMu[2]), '-r')
        plt.plot(WspBinCenters, np.ones(WspBinCenters.shape)*SigmaSigmaRef, '-g')# just a constant
        plt.savefig(cur + '\\res\\bin_center_plot_2.eps')
    # %% PLOTTING CONDITION DISTRIBUTION OF TURBULENCE 
    
    if turbulence_plot:   
        # What we want to do here....
        #
        # We have a cloud of points for a graph of mu and sigma mu
        # If you start at a particular windspeed (this is a slice of the graph), based on your data points you can look how your turbulence looks
        # You create a normal distribution at that windspeed
        # You can calculate mean for every slice , getting multiple points over the curve, calculating mean wind speed
        # Simple way to build this graph: create bins, get the mean and std-dev per bin, than you can add a polynomial, creating a continuous curve
        # This is what we are doing in this cell
        
        
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
        SigmaSigmaFunc = lambda u: np.ones(u.shape)*SigmaSigmaRef
        
        # plotting with newly created functions
        
        plt.plot(WspBinCenters, MuSigmaBinned, "-ok")
        plt.plot(WspBinCenters, SigmaSigmaBinned, "-xb") # for turbulence this is a common assumption
        plt.plot(WspBinCenters, MuSigmaFunc(WspBinCenters), '-r')
        plt.plot(WspBinCenters, SigmaSigmaFunc(WspBinCenters), '-g')# just a constant
        plt.savefig(cur + '\\res\\turb_plot.eps')
    
    if monte_carlo_plot:
        NMC = 10000
        Fwind = np.random.rand(NMC)
        Urand = stats.weibull_min.ppf(Fwind, scale = WeibullA, c = Weibullk)
        plt.hist(Urand,30)
        
        # Now do sigma 
    
        FsigmaU = np.random.rand(NMC)
        MuSigmaU = MuSigmaFunc(Urand)
        
        plt.plot(Urand, MuSigmaU, "xk")
        plt.savefig(cur + '\\res\\monte_carlo_plot.eps')
        plt.show()
        
        # new thing will create straight line
    
        SigmaSigmaU = SigmaSigmaFunc(Urand)
        
        plt.plot(Urand, SigmaSigmaU, "xk")
        plt.savefig(cur + '\\res\\monte_carlo_2_plot.eps')
        plt.show()
        
        # use functions all the way above
    
    
        SigmaUrand = LogNormDist(2,FsigmaU, MuSigmaU, SigmaSigmaU)
        
        
        plt.plot(Urand, SigmaUrand, "xk")
        plt.savefig(cur + '\\res\\monte_carlo_3_plot.eps')
        plt.show()
        
        # PLOT TURBULENCE INCLUDING DISTRIBUTION PARAMETERS
    
        fig,ax = plt.subplots(1,2, figsize = (12,5))
        ax[0].plot(WindData['Wsp'],WindData['SigmaU'],'xk')
        ax[0].plot(WspBinCenters,MuSigmaBinned,'-r')
        ax[0].plot(WspBinCenters,SigmaSigmaBinned,'-b')
        ax[0].plot(WspBinCenters,MuSigmaFunc(WspBinCenters),'--g')
        ax[0].set_xlim([0,40])
        ax[0].set_ylim([0,7])
        ax[1].plot(Urand,SigmaUrand,'*b')
        ax[1].set_xlim([0,40])
        ax[1].set_ylim([0,7])
        plt.savefig(cur + '\\res\\turbulence_plot.eps')
        plt.show()
    
