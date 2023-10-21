# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:31:40 2023

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

from datetime import datetime

# %% IMPORT DATA

WindData = pd.read_csv('HovsoreData_Sonic_100m_2004-2013.csv')

# Filter rows where 'Wsp' is less than or equal to 35 m/s
filtered_WindData = WindData[WindData['Wsp'] <= 35]

filtered_WindData.loc[filtered_WindData["TI"] <= 0.001] = np.nan

filtered_WindData = filtered_WindData.dropna()

# Save the filtered data to a new CSV file
filtered_csv_file = 'FilteredWindData.csv'
filtered_WindData.to_csv(filtered_csv_file, index=False)

WindData = filtered_WindData

U = WindData['Wsp']

#%% U MEAN & U STD

Umean = np.mean(U)
Ustd = np.std(U)

n = len(U) # Count the number of samples


print("Umean = ", Umean)
print("Ustd = ", Ustd)
print("n = ", n)

# %% BOOTSTRAP
alpha =  1 - 0.95# Corresponding to 95% probability  ( alpha = 1-p)


now = datetime.now()

start_time = now.strftime("%H:%M:%S")

print("Start of bootstrap: ", start_time)

Nbootstrap = 100 #99999  # How many times you wanna do this random act
#BootstrapSize = len(U)
#BootstrapMeans = np.mean(U)

N1year = int(1*365*24*6) # taking an interval for 1 year with 365 days with 24 hours with 6 times 10 minute intervals


#for i in range(N1year): # you take the range of this 1 year
    
# here you take a random sample and then use that sample to locate the next year
Yearsample = np.random.randint(low = 0, high = len(WindData) - N1year) #, size = (Nbootstrap,BootstrapSize))

print("Yearsample = ", Yearsample)

Winddata_1_year = WindData[Yearsample:Yearsample + N1year]
print("Winddata_1_year = ", Winddata_1_year)

#Bsample = np.random.randint(low = 0, high = 8, size = (Nbootstrap,BootstrapSize))
#Bsample = np.random.randint(low = 0, high = len(WindData) - N1year)

BootstrapSize = len(Winddata_1_year)

# now select the year in which we do the bootstrapping
BootstrapSample = np.zeros((Nbootstrap,BootstrapSize))

# now do bootstrapping over the chosen year
#for j in range(Nbootstrap):
#    # Extract a bootstrap sample using the indices from Bsample
#    bootstrap_indices = Bsample[j] + np.arange(BootstrapSize)
#    BootstrapSample[j, :] = Winddata[bootstrap_indices]
print("========================================================")
print("Starting Bootstrap...")
for i in range(Nbootstrap):
    print("i = ", i)
    for j in range(BootstrapSize):
        ##print("Winddata_1_year['Timestamp'].idxmin() = ",Winddata_1_year['Timestamp'].idxmin())
        ##print("Winddata_1_year['Timestamp'].idxmax() = ",Winddata_1_year['Timestamp'].idxmax())
        #Bsample = np.random.randint(low = Winddata_1_year['Timestamp'].idxmin(), high =Winddata_1_year['Timestamp'].idxmax())
        #print("Bsample = ",Bsample)
        
                # Initialize Bsample to be outside the valid range
        Bsample = np.random.randint(low = Winddata_1_year['Timestamp'].idxmin(), high =Winddata_1_year['Timestamp'].idxmax())
        #Bsample = -1
        
        # Keep generating a new random sample until it's a valid index
        while Bsample not in Winddata_1_year.index:
            #print("Wrong sample, i = ",i,", j = ", j)
            Bsample = np.random.randint(low=Winddata_1_year['Timestamp'].idxmin(), high=Winddata_1_year['Timestamp'].idxmax())
        
        ##print("Bsample =", Bsample)
        
        #print("Winddata_1_year = ", Winddata_1_year)
        #print(Winddata_1_year['Wsp', index = Bsample])
        #print(Winddata_1_year)
        #print(Winddata_1_year[Winddata_1_year['Timestamp'].idxmax()])
        ##print("Winddata_1_year.loc[Bsample, 'Wsp'] = ", Winddata_1_year.loc[Bsample, 'Wsp'])
        BootstrapSample[i,j] = Winddata_1_year.loc[Bsample, 'Wsp'] #Winddata_1_year['Wsp', index = Bsample]

print("DONE!!")
# repeat this process

#BootstrapSample[i,j] = U[Bsample[i,j]]
        
#BootstrapSample = 

BootstrapMeans = np.mean(BootstrapSample, axis=1)
BootstrapMeans = np.sort(BootstrapMeans)

Rlow = int((Nbootstrap+1) * (alpha/2) )
Rhigh = int((Nbootstrap+1) * (1-alpha/2))

CIn_B = BootstrapMeans[Rlow]
CIp_B = BootstrapMeans[Rhigh]

print('Confidence interval based on bootstrapping: [' + str(CIn_B) + ', ' + str(CIp_B) + ']')


now = datetime.now()

end_time = now.strftime("%H:%M:%S")

print("End of bootstrap: ", end_time)

print("+++++++++++++++++++++++++++++++++++++++++++++")
#print("Duration of bootstrap: ", int(end_time.hour)-int(start_time.hour),":",int(end_time.minute)-int(start_time.minute),":",int(end_time.second)-int(start_time.second))


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
ax0.set_xticks([1,2,3])
ax0.set_xticklabels(['Normal dist.','Bootstrapping'])
ax0.set_ylabel('Annual mean wind speed [m/s]')
plt.show()

# Plot pdfs

Ubins = np.linspace(8.5,12,100)

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
plt.xlabel('Annual mean wind speed [m/s]')
plt.ylabel('Probability density')
plt.legend()

plt.show()

