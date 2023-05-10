#!/usr/bin/env python
# coding: utf-8

# # FFT Base Model
# Purpose:
# - Simulating good data to mimic a sinusoidal pattern for heart rate
# - Using this data to train the ANN model
# 
# Notes:
# - no downsampling is applied
# - only realistically applicable for heart rate, as Dr.Qazi confirmed a sinusoidal pattern

# # Imports

# In[ ]:


# import libraries
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import openpyxl
import pandas as pd
import sklearn
import random
from sklearn.preprocessing import normalize
from scipy.fft import fft, fftfreq, ifft


# # Loading/Preprocessing

# In[ ]:


#Importing the data using an environment variable

csv_folder_path = os.environ.get("CSV_Folder_Path")
file_path = os.path.join(csv_folder_path, "uq_vsd_case04_alldata.csv")


# In[ ]:


# Filtering the data
df = pd.read_excel(file_path, index_col=False, error_bad_lines=False, usecols=['Time', 'HR', 'SpO2', 'NBP (Mean)', 'ECG'])

df = df.dropna() #drop rows with NaN

case4_filteredData = df.loc[(~df['HR'].isnull()) & (~df['SpO2'].isnull()) & (~df['NBP (Mean)'].isnull()) & (~df['ECG'].isnull())]
case4_filteredData.drop('Time', axis=1, inplace=True)
print(case4_filteredData.head(10))
print(case4_filteredData.tail(10))
print(case4_filteredData.shape)


# In[ ]:


# plot data
dataset = case4_filteredData # use if google drive
values = dataset.values

# specify columns to plot
groups = [0,1,2,3]
i = 1
for g in groups:
  plt.subplot(len(groups), 1, i)
  plt.plot(values[:, g])
  plt.title(dataset.columns[g], y=0.5, loc='right')
  i += 1
# plt.show()


# # FFT Forecasting

# In[ ]:


# Prepping the data
raw_var_data = case4_filteredData['HR'] # replace column name with variable you want


# In[ ]:


all_modified_data = np.copy(np.array(raw_var_data))

# looping to fill desired time
for j in range(0, 10):
  # Applying fft
  original_data = np.copy(np.array(raw_var_data))
  fft_sig = fft(np.array(raw_var_data))

  # leaving first two values of fft signal
  input_fft_sig = np.copy(fft_sig[2:])

  # taking absolute value since signal is complex numbers
  abs_sig = np.absolute(input_fft_sig)

  # getting indices of 20 highest values
  sorted_index_array = np.argpartition(abs_sig, -20)[-20:]

  # modifying maximum values by a randomized percentage
  percentage = random.randint(80,100)/100
  # percentage = 0.5 # testing
  print(percentage)
  for i in range(0, len(sorted_index_array) - 1):
    index = sorted_index_array[i]
    fft_sig[index+2] = fft_sig[index+2] * percentage # fixing offset of 2

  # Applying inverse fft
  inversed_sig = np.real(ifft(fft_sig))

  all_modified_data = np.append(all_modified_data, inversed_sig)


# In[ ]:


# Plotting

plt.plot(all_modified_data, 'b', label='Modified')
plt.plot(original_data, 'r', label='Original')
plt.title("FFT Base Model - Heart Rate (HR)")
plt.legend()
plt.figure(figsize=(20, 8))
plt.show()

