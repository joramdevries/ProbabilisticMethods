# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:56:59 2023

@author: joram
"""
# %% IMPORT LIBRARIES

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network
import os

cur = os.getcwd()

from datetime import datetime, timedelta


# %% BOOTSTRAP

Nbootstrap = 1000

def bootstrap_function(WindData, Nbootstrap, plots):
    now = datetime.now()
    
    start_time_str = now.strftime("%H:%M:%S")
    
    start_time = datetime.strptime(start_time_str, "%H:%M:%S")
    
    print("Start of bootstrap: ", start_time_str)
    
     #99999  # How many times you wanna do this random act
    #BootstrapSize = len(U)
    #BootstrapMeans = np.mean(U)
    
    alpha =  1 - 0.95# Corresponding to 95% probability  ( alpha = 1-p)
    
    N1year = int(1*365*24*6) # taking an interval for 1 year with 365 days with 24 hours with 6 times 10 minute intervals
    
    
    BootstrapSize = N1year
    
    # now select the year in which we do the bootstrapping
    BootstrapSample = np.zeros(Nbootstrap)
    
    #years = ["2004","2005","2006","2007","2008","2009","2010","2011","2012","2013"]
    print("========================================================")
    print("Starting Bootstrap...")

    for n in range(Nbootstrap):
        
        Bsample = np.random.randint(low = 0, high = len(WindData) - N1year)
        
        # Use .iloc to access the row by index
        start_year = WindData['Timestamp'].iloc[Bsample]
        # Add one year to the start_year
        end_year = start_year + timedelta(days=365)
        
        #while end_year not in WindData['Timestamp']:
        #    print("End year not an option")
        #    end_year -= end_year 
        
        #print("Start year:", start_year)
        #print("End year:", end_year)
        
        random_year = WindData.loc[(WindData['Timestamp'] >= start_year) & (WindData['Timestamp'] <= end_year)]
        
        year_wsp_mean = np.mean(random_year['Wsp'])
        

        #print("random_year: ", random_year["Timestamp"])
        #print("year_wsp_mean: ", year_wsp_mean)
        #print("-------------------------------------------------------------")   
        
        
        #s = np.random.normal(year_wsp_mean, year_wsp_mean*0.01, 1000)
        
        #Bsample_final = np.random.choice(s)
        
        BootstrapSample[n] = year_wsp_mean #Bsample_final
        
    print("DONE!!")
    
    if plots:
        # Create a histogram
        plt.hist(BootstrapSample, bins=20, edgecolor='black')
        plt.title('Bootstrap Sample Histogram')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.savefig(cur + '\\res\\Histogram_Bootstrap.eps')
        plt.show()

    mean_bsample = BootstrapSample.mean()
    
    if plots:
        # Create a histogram
        plt.hist(BootstrapSample/mean_bsample, bins=20, edgecolor='black')
        plt.title('Bootstrap Sample Histogram')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.savefig(cur + '\\res\\Histogram_Bootstrap_Uncertainty.eps')
        plt.show()

    BootstrapSample_new = BootstrapSample/mean_bsample

    # fit lognormal distribution
    shape, loc, scale = stats.lognorm.fit(BootstrapSample_new, loc=0)
    pdf_lognorm = stats.lognorm.pdf(BootstrapSample_new, shape, loc, scale)

    if plots:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax2 = ax.twinx()
    
        ax.hist(BootstrapSample_new, bins='auto', density=True)
        ax2.hist(BootstrapSample_new, bins='auto')
        ax.plot(BootstrapSample_new, pdf_lognorm)
        ax2.set_ylabel('Frequency')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Xw')
        ax.set_title('Bootstrap Sample Histogram')
    
        plt.savefig(cur + '\\res\\Histogram_Bootstrap_Uncertainty.eps')
        plt.show()

    BootstrapMeans = np.sort(BootstrapSample)

    Rlow = int((Nbootstrap+1) * (alpha/2) )
    #print(Rlow)
    Rhigh = int((Nbootstrap+1) * (1-(alpha/2)))
    #print(Rhigh)

    CIn_B = BootstrapMeans[Rlow]
    CIp_B = BootstrapMeans[Rhigh]


    print('Confidence interval based on bootstrapping: [' + str(CIn_B) + ', ' + str(CIp_B) + ']')


    now = datetime.now()

    end_time_str = now.strftime("%H:%M:%S")

    end_time = datetime.strptime(end_time_str, "%H:%M:%S")

    print("End of bootstrap: ", end_time_str)

    print("+++++++++++++++++++++++++++++++++++++++++++++")
    mean_of_sample = BootstrapMeans.mean()
    print("mean_of_sample: ", mean_of_sample)
    std_of_sample = BootstrapMeans.std()
    
    new_std = np.sqrt(std_of_sample**2 + 0.01**2 + 0.01**2)
    print("std_of_sample: ", std_of_sample)
    print("new_std: ", new_std)
    X_w = [mean_of_sample/mean_of_sample, new_std/mean_of_sample]
    print("X_w: ", X_w)
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    # Calculate duration
    duration = end_time - start_time

    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    print(f"Duration of bootstrap: {hours} hours, {minutes} minutes, {seconds} seconds")
    
    return BootstrapMeans, BootstrapSample, X_w

# %% MAIN LOOP
if __name__ == "__main__":
    
    # %% IMPORT DATA

    WindData = pd.read_csv('HovsoreData_Sonic_100m_2004-2013.csv')

    WindData['Timestamp'] = pd.to_datetime(WindData['Timestamp'], format='%Y%m%d%H%M')

    WindData['Year'] = WindData['Timestamp'].dt.year

    print("Original Data has length of ", len(WindData))

    # %% BOOTSTRAP
    alpha =  1 - 0.95# Corresponding to 95% probability  ( alpha = 1-p)


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

    #%% U MEAN & U STD

    Umean = np.mean(U)
    Ustd = np.std(U)

    n = len(U) # Count the number of samples


    print("Umean = ", Umean)
    print("Ustd = ", Ustd)
    print("n = ", n)
    
    
    plots = True
    
    BootstrapMeans, BootstrapSample, X_w = bootstrap_function(WindData, Nbootstrap, plots)
    
    #Rlow = int((Nbootstrap+1) * (alpha/2) )
    #print(Rlow)
    #Rhigh = int((Nbootstrap+1) * ((1-alpha)/2))
    #print(Rhigh)
    
    #CIn_B = BootstrapMeans[Rlow]
    #CIp_B = BootstrapMeans[Rhigh]
    
    n_bootstrap_samples = len(BootstrapMeans)
    Rlow = int(n_bootstrap_samples * alpha / 2)
    Rhigh = int(n_bootstrap_samples * (1 - alpha / 2))
    CIn_B = BootstrapMeans[Rlow]
    CIp_B = BootstrapMeans[Rhigh]
    
    print(Rlow)
    print(Rhigh)
    print('Confidence interval based on bootstrapping: [' + str(CIn_B) + ', ' + str(CIp_B) + ']')
    
    
    
    # %% NORMAL DISTRIBUTION
    
    # Confidence intervals using directly the Standard Normal distribution
    k_alpha = stats.norm.ppf(alpha/2)
    k_alpha_p = - stats.norm.ppf(1-alpha/2)
    
    CIn_N = Umean + k_alpha * (Ustd/(np.sqrt(n)))
    CIp_N = Umean - k_alpha * (Ustd/(np.sqrt(n)))
    
    print('Confidence interval based on the Normal distribution: [' + str(CIn_N) + ', ' + str(CIp_N) + ']')
    
    
    # %% PLOTTING
    
    # Plot errorbars
    fig0, ax0 = plt.subplots()
    ax0.errorbar([1, 2], [Umean, np.mean(BootstrapMeans)],
                 yerr = [(CIp_N - CIn_N), (CIp_B - CIn_B)],
                linestyle = '',marker = 'o',capsize = 5)
    ax0.set_xlim([0.5,3.5])
    ax0.set_xticks([1,2])
    ax0.set_xticklabels(['Normal dist.','Bootstrapping'])
    ax0.set_ylabel('Annual mean wind speed [m/s]')
    plt.savefig(cur + '\\res\\Normal_vs_Bootstrap.eps')
    plt.show()
    
    # Plot pdfs
    
    Ubins = np.linspace(7.5,12,100)
    
    pdf_N = stats.norm.pdf(Ubins,Umean,Ustd/np.sqrt(n))
    dU = Ubins[1]-Ubins[0] # Scaling factor for the t-pdf to make sure we get a valid pdf for every bin spacing
    pdf_T = (1/np.sqrt(dU))*stats.t.pdf((Ubins - Umean)/(Ustd/np.sqrt(n)), n - 1)
    
    # Generating an empirical pdf from the bootstrap sample
    BootstrapHist = np.histogram(BootstrapMeans,bins = Ubins)
    
    #plt.hist(BootstrapMeans,bins = Ubins) 
    
    BootstrapDist = stats.rv_histogram(BootstrapHist)
    pdf_B = BootstrapDist.pdf(Ubins)
    
    fig1, ax1 = plt.subplots()
    p11 = ax1.plot(Ubins,pdf_N,'--r', label = 'Normal')
    p12 = ax1.plot(Ubins,pdf_B,'-b', label = 'Bootstrapping')
    plt.xlabel('Annual mean wind speed [m/s]')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(cur + '\\res\\Normal_vs_Bootstrap_Probability_Density.eps')
    
    plt.show()
    
    # Plot pdfs
    
    Ubins = np.linspace(7.5,10,100)
    
    pdf_N = stats.norm.pdf(Ubins,Umean,Ustd/np.sqrt(n))
    dU = Ubins[1]-Ubins[0] # Scaling factor for the t-pdf to make sure we get a valid pdf for every bin spacing
    pdf_T = (1/np.sqrt(dU))*stats.t.pdf((Ubins - Umean)/(Ustd/np.sqrt(n)), n - 1)
    
    # Generating an empirical pdf from the bootstrap sample
    BootstrapHist = np.histogram(BootstrapMeans,bins = Ubins)
    BootstrapDist = stats.rv_histogram(BootstrapHist)
    pdf_B = BootstrapDist.pdf(Ubins)
    
    fig1, ax1 = plt.subplots()
    p11 = ax1.plot(Ubins,pdf_N,'--r', label = 'Normal')
    p12 = ax1.plot(Ubins,pdf_B,'-b', label = 'Bootstrapping')
    #p13 = ax1.plot(Ubins,pdf_T,'-k', label = 'T-dist')
    plt.xlabel('Annual mean wind speed [m/s]')
    plt.ylabel('Probability density')
    plt.legend()
    
    plt.savefig(cur + '\\res\\Normal_vs_Bootstrap_Probability_Density_withT.eps')
    
    plt.show()
    
    #%% PLOT EXTRA
    
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].plot(Ubins, pdf_N, '--r', label='Normal')
    ax1[0].plot(Ubins, pdf_B, '-b', label='Bootstrapping')
    ax1[0].set_xlabel('Annual mean wind speed [m/s]')
    ax1[0].set_ylabel('Probability density')
    ax1[0].legend()
    
    ax1[1].hist(BootstrapMeans, 100, label='Bootstrapping')
    ax1[1].set_xlabel('Annual mean wind speed [m/s]')
    ax1[1].set_ylabel('Frequency')
    plt.show()
    
    
    
    
    # %% YEAR MEANS PLTS
    
    WindData['Year'] = WindData['Timestamp'].dt.year
    
    years = ["2004","2005","2006","2007","2008","2009","2010","2011","2012","2013"]
    
    U_YMP = np.zeros(len(years))
    
    for y in range(len(years)):
        
        Winddata_1_year = WindData[WindData['Year'] == int(years[y])]
        
        U_YMP[y] = np.mean(Winddata_1_year['Wsp'])
        
    #U = [8.97, 8.56, 9.11, 8.79, 8.27, 8.40, 9.56, 8.02]
    
    U_YMP_mean = np.mean(U_YMP)
    U__YMP_std = np.std(U_YMP)
    
    alpha =  1 - 0.95# Corresponding to 95% probability  ( alpha = 1-p)
    
    n = len(U_YMP) # Count the number of samples

    Nbootstrap = 10000
    BootstrapSize = len(U_YMP)
    
    Bsample = np.random.choice(U_YMP, size = (BootstrapSize,Nbootstrap))
    BootstrapMeans = np.sort(Bsample.mean(0))
    print(np.shape(BootstrapMeans))
    print(Bsample.mean(0))
    
    Rlow = int(np.around(Nbootstrap*alpha/2))
    Rhigh = int(np.around((1-alpha/2)*Nbootstrap))
    
    CIn_B = BootstrapMeans[Rlow]
    CIp_B = BootstrapMeans[Rhigh]
    
    print('Confidence interval based on bootstrapping: [' + str(CIn_B) + ', ' + str(CIp_B) + ']')
    
    #%% PLOT EXTRA
    
    Ubins = np.linspace(7.5,12,100)
    
    pdf_N = stats.norm.pdf(Ubins,U_YMP_mean,U__YMP_std/np.sqrt(n))
    dU = Ubins[1]-Ubins[0] # Scaling factor for the t-pdf to make sure we get a valid pdf for every bin spacing
    pdf_T = (1/np.sqrt(dU))*stats.t.pdf((Ubins - U_YMP_mean)/(U__YMP_std/np.sqrt(n)), n - 1)
    
    # Generating an empirical pdf from the bootstrap sample
    BootstrapHist = np.histogram(BootstrapMeans,bins = Ubins)
    
    #plt.hist(BootstrapMeans,bins = Ubins) 
    
    BootstrapDist = stats.rv_histogram(BootstrapHist)
    pdf_B = BootstrapDist.pdf(Ubins)
    
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].plot(Ubins, pdf_N, '--r', label='Normal')
    ax1[0].plot(Ubins, pdf_B, '-b', label='Bootstrapping')
    ax1[0].set_xlabel('Annual mean wind speed [m/s]')
    ax1[0].set_ylabel('Probability density')
    ax1[0].legend()
    
    ax1[1].hist(BootstrapMeans, 100, label='Bootstrapping')
    ax1[1].set_xlabel('Annual mean wind speed [m/s]')
    ax1[1].set_ylabel('Frequency')
    
    plt.savefig(cur + '\\res\\correct_Bootstrap.eps')
    
    plt.show()
    
    
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    mean_of_sample = BootstrapMeans.mean()
    print("mean_of_sample: ", mean_of_sample)
    std_of_sample = BootstrapMeans.std()
    
    new_std = np.sqrt((std_of_sample/mean_of_sample)**2 + 0.01**2 + 0.01**2)
    print("std_of_sample: ", std_of_sample)
    print("new_std: ", new_std)
    X_w = [mean_of_sample/mean_of_sample, new_std]
    print("X_w: ", X_w)
    print("+++++++++++++++++++++++++++++++++++++++++++++")