 # install external libraries
!pip install wfdb
!pip install tensorflow_addons
!pip install keras-tuner

# import libraries
import torch
import keras_tuner as kt
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import tensorflow as tf
import tensorflow_addons as tfa
import time
import wfdb
from wfdb import processing

# Function for splitting data into training and test set non-randomly

# This is from the 3 variables model 

"""
This functions splits the data into training and test sets.

@param data_df : a Pandas Dataframe that contains the data to be split into training and test sets
@param train_size : a double/float that is within the range of 0 and 1 and represents the fraction of the data to be used for the training set

@returns two Numpy arrays that represent the training and test sets, respectively
"""
def split_data_train_test(data_df, train_size):
  train_df, test_df = data_df[0:round(train_size * len(data_df)), :], data_df[round(train_size * len(data_df)):len(data_df), :]

  return train_df, test_df

# Function for scaling the data
"""
This function takes in the data to be scaled and uses Min-Max Scaling.

@param data : a Numpy array that represents the inputted data to be scaled

@returns the scaled data as a Numpy array, the MinMax scaler object, the scaling factor used, and the minimum value of the original inputted data
"""
def minMaxScaling(data):
  scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0.1, 1))
  scaler.fit(data)

  scaleFactor = scaler.scale_
  data_min = scaler.data_min_
  data_max = scaler.data_max_
  data = scaler.transform(data)

  return data, scaler, scaleFactor, data_min, data_max

"""
This function takes in the data to be scaled and uses Standard Scaling.

@param data : a Numpy array that represents the inputted data to be scaled

@returns the scaled data as a Numpy array, the Standard scaler object, the mean of the original inputted data, and the standard deviation of the original inputted data
"""
def stdScaling(data):
  scaler = sklearn.preprocessing.StandardScaler()
  scaler.fit(data)

  mean = scaler.mean_
  std_dev = scaler.var_ ** 0.5
  data = scaler.transform(data)

  return data, scaler, mean, std_dev

"""
This function plots the inputted data in a 2D scatterplot.

@param xvals : a 1D Numpy array that contains the data to be plotted on the x-axis
@param data : a 1D Numpy array that contains the data to be plotted on the y-axis
@param xVariable : a string that represents the label of the data to be plotted for x-axis
@param yVariable : a string that represents the label of the data to be plotted for y-axis
"""
def plot_data(xvals, data, xVariable, yVariable):
  plt.figure(figsize=(10, 5))
  plt.scatter(xvals, data)
  plt.title("Plot of Data for " + yVariable + ", " + xVariable)
  plt.ylabel(yVariable)
  plt.xlabel(xVariable)
  plt.show()

def downsample_data(window):
  df_temp = pd.DataFrame(columns=['HR', 'SpO2', 'NBP (Mean)', 'ECG'])
  HR_sum = 0
  SPO_sum = 0
  NBP_sum = 0
  ECG_sum = 0

  for i in range(1, len(values) - 1):
    HR_sum += values[i, 0]
    SPO_sum += values[i, 1]
    NBP_sum += values[i, 2]
    ECG_sum += values[i, 3]
    if (i + 1) % window == 0:
      HR_sum = HR_sum/window
      SPO_sum = SPO_sum/window
      NBP_sum = NBP_sum/window
      ECG_sum = ECG_sum/window
      row = {'HR':HR_sum, 'SpO2':SPO_sum, 'NBP (Mean)':NBP_sum, 'ECG':ECG_sum}
      df_temp = df_temp.append(row, ignore_index = True)
      HR_sum = 0
      SPO_sum = 0
      NBP_sum = 0
      ECG_sum = 0
  return df_temp

  # Function for splitting data into training and test set non-randomly

# This is from the 3 variables model 

"""
This functions splits the data into training and test sets.

@param data_df : a Pandas Dataframe that contains the data to be split into training and test sets
@param train_size : a double/float that is within the range of 0 and 1 and represents the fraction of the data to be used for the training set

@returns two Numpy arrays that represent the training and test sets, respectively
"""
def split_data_train_test(data_df, train_size):
  train_df, test_df = data_df[0:round(train_size * len(data_df)), :], data_df[round(train_size * len(data_df)):len(data_df), :]

  return train_df, test_df

# Function for scaling the data
"""
This function takes in the data to be scaled and uses Min-Max Scaling.

@param data : a Numpy array that represents the inputted data to be scaled

@returns the scaled data as a Numpy array, the MinMax scaler object, the scaling factor used, and the minimum value of the original inputted data
"""
def minMaxScaling(data):
  scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0.1, 1))
  scaler.fit(data)

  scaleFactor = scaler.scale_
  data_min = scaler.data_min_
  data_max = scaler.data_max_
  data = scaler.transform(data)

  return data, scaler, scaleFactor, data_min, data_max

"""
This function takes in the data to be scaled and uses Standard Scaling.

@param data : a Numpy array that represents the inputted data to be scaled

@returns the scaled data as a Numpy array, the Standard scaler object, the mean of the original inputted data, and the standard deviation of the original inputted data
"""
def stdScaling(data):
  scaler = sklearn.preprocessing.StandardScaler()
  scaler.fit(data)

  mean = scaler.mean_
  std_dev = scaler.var_ ** 0.5
  data = scaler.transform(data)

  return data, scaler, mean, std_dev

"""
This function plots the inputted data in a 2D scatterplot.

@param xvals : a 1D Numpy array that contains the data to be plotted on the x-axis
@param data : a 1D Numpy array that contains the data to be plotted on the y-axis
@param xVariable : a string that represents the label of the data to be plotted for x-axis
@param yVariable : a string that represents the label of the data to be plotted for y-axis
"""
def plot_data(xvals, data, xVariable, yVariable):
  plt.figure(figsize=(10, 5))
  plt.scatter(xvals, data)
  plt.title("Plot of Data for " + yVariable + ", " + xVariable)
  plt.ylabel(yVariable)
  plt.xlabel(xVariable)
  plt.show()

# Splitting data into previous values and next values
# From the 3 variables model.
"""
This function splits the training and test sets and turns a certain number of previous values as "features" and the next value as the "target"

@param train_array : a Numpy array with at least 2 columns of the training set values
@param test_array : a Numpy array with at least 2 columns of the test set values
@param numOfPrevValues : an integer that represents the number of previous rows of values to use as features
@param targetIndex : an integer that represents the column number to use for target values

@returns four Numpy arrays for the previous values in the training set, the next 1 value in the training set, the previous values
         in the test set, and the next 1 value in the test set
"""
def splitDataToFeaturesTargetValues(train_array, test_array, numOfPrevValues=27, targetIndex=1):

  prevData_train = []
  nextData_train = [] 
  prevData_test = []
  nextData_test = []

  for i in range(numOfPrevValues, len(train_array)):
    prevData_train.append(train_array[i-numOfPrevValues: i])
    nextData_train.append(train_array[i][targetIndex])

  for j in range(numOfPrevValues, len(test_array)):
    prevData_test.append(test_array[j-numOfPrevValues: j])
    nextData_test.append(test_array[j][targetIndex])
  
  prevData_train = np.array(prevData_train)
  nextData_train = np.array(nextData_train).reshape(len(nextData_train), 1)
  prevData_test = np.array(prevData_test)
  nextData_test = np.array(nextData_test).reshape(len(nextData_test), 1)

  return prevData_train, nextData_train, prevData_test, nextData_test

# Function that makes the loss function plots and linear regression model
# From the 3 Variables Model "Mostly not needed"

"""
This function plots a loss function based on the output of the model fitting.

@param loss_vals : a Tensorflow History object from model fitting, which is used to get the training and validation loss values
@param startRange : an integer that is >= 0, is within the indices of loss_vals, and represents the start indices of the loss values to plot
@param endRange : an integer that is <= len(loss_vals) - 1, is larger than startRange, and represents the end indices of the loss values to plot
"""
def plot_loss(loss_vals, startRange, endRange):
  plt.figure(figsize=(12, 8))
  plt.plot(loss_vals.history['loss'][startRange:endRange], label="Training Loss") 
  plt.plot(loss_vals.history['val_loss'][startRange:endRange], label="Validation Loss") 
  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.title("Learning Curve")
  plt.legend()

"""
This function plots the inputted actual and predicted data in a 2D scatterplot.

@param xvals : a 1D Numpy array that contains the data to be plotted on the x-axis
@param actual_data : a 1D Numpy array that contains the actual data to be plotted on the y-axis
@param predicted_data : a 1D Numpy array that contains the predicted data to be plotted on the y-axis
@param xName : a string that represents the x-values and label for the x-axis
@param yName : a string that represents the y-values and label for the y-axis
"""
def plot_model_data(xvals, actual_data, predicted_data, xName, yName):
  plt.figure(figsize=(12, 8))
  plt.scatter(xvals, actual_data, label="Actual Data", color="r")
  plt.plot(xvals, predicted_data, label="Predicted Data", color="black")
  plt.title("Plot of Model Data for " + xName + " Against " + yName)
  plt.xlabel(xName)
  plt.ylabel(yName)
  plt.legend()

"""
This function plots the inputted actual and predicted data in a 3D scatterplot.

@param x_val : a 1D Numpy array that contains the data to be plotted on the x-axis
@param y_val : a 1D Numpy array that contains the data to be plotted on the y-axis
@param actual_data : a 1D Numpy array that contains the actual data to be plotted on the z-axis
@param predicted_data : a 1D Numpy array that contains the predicted data to be plotted on the z-axis
@param xName : a string that represents the x-values and label for the x-axis
@param yName : a string that represents the y-values and label for the y-axis
@param zName : a string that represents the z-values and label for the z-axis
"""
def plot_model_data3D(x_val, y_val, actual_data, predicted_data, xName, yName, zName):
  fig = plt.figure(figsize=(12, 8))

  ax = plt.axes(projection='3d')
  ax.scatter3D(x_val, y_val, actual_data, label='Actual Data', color="red")
  ax.plot3D(x_val, y_val, predicted_data, label='Predicted Data', color="black")

  ax.set_title("Plot of Model Data for " + xName + " And " + yName + " Against " + zName, pad=20)
  ax.set_xlabel(xName)
  ax.set_ylabel(yName)
  ax.set_zlabel(zName)
  ax.legend()

#Need to allow code to access the CSV Dataset. 
#Setting up an environment variable. Will add a readme file in order to show how to use this method.


params: csv_file_name, new_data_file_name

csv_folder_path = os.environ.get("CSV_Folder_Path")
case4_alldata_file_path = os.path.join(csv_folder_path, csv_file_name)

# FILTERING THE DATA
df = pd.read_csv(case4_alldata_file_path, index_col=False, error_bad_lines=False, usecols=['Time', 'HR', 'SpO2', 'NBP (Mean)', 'ECG'])

df = df.dropna() #drop rows with NaN

case4_filteredData = df.loc[(~df['HR'].isnull()) & (~df['SpO2'].isnull()) & (~df['NBP (Mean)'].isnull()) & (~df['ECG'].isnull())]
case4_filteredData.drop('Time', axis=1, inplace=True)
print(case4_filteredData.head(10))
print(case4_filteredData.tail(10))
print(case4_filteredData.shape)

new_data_file_path = os.path.join(csv_folder_path, new_data_file_name)

dataset = pd.read_csv(new_data_file_path, index_col=0, header=0)
values = dataset.values

# specify columns to plot
groups = [0,1,2,3]
i =1
for g in groups:
  plt.subplot(len(groups), 1, i)
  plt.plot(values[:, g])
  plt.title(dataset.columns[g], y=0.5, loc='right')
  i += 1
plt.show()

old_data = case4_filteredData
print('Size of old data:')
print(old_data.shape)

# downsampling data
case4_filteredData = downsample_data(15)

print('Size of new data:')
print(case4_filteredData.shape)

# plot data
dataset = case4_filteredData # use if google drive
values = dataset.values

# specify columns to plot
groups = [0,1,2,3]
i =1
for g in groups:
  plt.subplot(len(groups), 1, i)
  plt.plot(values[:, g])
  plt.title(dataset.columns[g], y=0.5, loc='right')
  i += 1
# plt.show()

# Scaling and splitting dataset, using R-Wave Voltages and SpO2 as features
pd.options.display.max_rows = 1000
print(case4_filteredData.head(20))


case4_filteredData_arr_scaled, minMaxScaler, minMaxScaleFactor, case4_filteredData_min, case4_filteredData_max = minMaxScaling(case4_filteredData)
# combinedData_arr_scaled, stdScaler, combinedData_mean, combinedData_stdDev = stdScaling(combinedData_df) # uncomment this if standard scaling will be used

# Add more columns names if more health variables are added.
case4_filteredData_scaled = pd.DataFrame(case4_filteredData_arr_scaled, columns=["HR", "SpO2","NBP","ECG"])

print()
print(case4_filteredData_scaled[0:20])

listOfVariables = ["HR", "SpO2","NBP","ECG"] # Add more columns names if more health variables are added.
listOfVariables_len = len(listOfVariables) - 1 

data_train, data_test = split_data_train_test(np.array(case4_filteredData_scaled[listOfVariables]), train_size=0.8) # scaled data
# data_train, data_test = split_data_train_test(np.array(combinedData_df[listOfVariables]), train_size=0.8) # non-scaled

print("\nOriginal data train and test: ")
print(data_train[0:20])

# print(" Original test")
print(data_test[0:20])


print(" Here to see the size of each data: ")
print("This is the size of the train data: ")
print(data_train.shape)
print("This is the size of the test data: ")
print(data_test.shape)

data_train_features, data_train_label, data_test_features, data_test_label = splitDataToFeaturesTargetValues(data_train, data_test, numOfPrevValues=50, targetIndex=1)
print("This is the data_train_features")
print(data_train_features[0:10])
print("This is the data_train_label")
print(data_train_label[0:10])
print("This is the data_test_features")
print(data_test_features[0:10])
print("This is the data_test_label")
print(data_test_label[0:10])
