#!/usr/bin/env python
# coding: utf-8

# # Latest LSTM Model to predict HR
# Purpose:
# -  Building LSTM (Long-Short Term Memory) Model using 4 health metrics to predict the same 4 health metrics simultaneously
# - utilizes many-to-many prediction method
# 
# Notes:
# - Data was downsampled
# - this model is similar to the latest ANN model in that it includes the predictions made when predicting the next set of values
# - Incomplete version as results are inadequate, needs modification 
# 

# # Installs and Imports

# In[ ]:


# install external libraries

get_ipython().system('pip install wfdb')
get_ipython().system('pip install tensorflow_addons')
get_ipython().system('pip install keras-tuner')


# In[ ]:


# import libraries

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


# # Load and preprocess data

# In[ ]:





# In[ ]:


#Importing the data using an environment variable

csv_folder_path = os.environ.get("CSV_Folder_Path")
file_path = os.path.join(csv_folder_path, "uq_vsd_case04_alldata.csv")


# In[ ]:


# FILTERING THE DATA

df = pd.read_csv(file_path, index_col=False, error_bad_lines=False, usecols=['Time', 'HR', 'SpO2', 'NBP (Mean)', 'ECG'])

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
i =1
for g in groups:
  plt.subplot(len(groups), 1, i)
  plt.plot(values[:, g])
  plt.title(dataset.columns[g], y=0.5, loc='right')
  i += 1
# plt.show()


# In[ ]:


# ----------------------------------------------------------------- MANUALLY RESAMPLE -------------------------------------------------------------------
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

old_data = case4_filteredData
print('Size of old data:')
print(old_data.shape)

# downsampling data
case4_filteredData = downsample_data(15)

print('Size of new data:')
print(case4_filteredData.shape)


# # Functions

# In[ ]:


# FUNCTIONS
"""
This functions splits the data into training and test sets.

@param data_df : a Pandas Dataframe that contains the data to be split into training and test sets
@param train_size : a double/float that is within the range of 0 and 1 and represents the fraction of the data to be used for the training set

@returns two Numpy arrays that represent the training and test sets, respectively
"""
def split_data_train_test(data_df, train_size):
  train_df, test_df = data_df[0:round(train_size * len(data_df)), :], data_df[round(train_size * len(data_df)):len(data_df), :]

  return train_df, test_df



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


"""
This functions gets and prints the root mean squared error (RMSE) of the targets.

@param test_targets : a Numpy array that represents the actual target values
@param predicted_targets : a Numpy array that represents the predicted target values

@returns the RMSE double/float value

MANUAL RMSE 
"""
def get_and_print_rmse_for_model(test_targets, predicted_targets):
  mse = sklearn.metrics.mean_squared_error(y_true=test_targets, y_pred=predicted_targets)  
  rmse = math.sqrt(mse)  
  print("RMSE: ", rmse)  
  
  return rmse 
"""
to calculate RMSE manually to add:
"""

#def calculate_rmse_manual(test_targets, predicted_targets):
  
"""
This functions gets and prints the coefficient of determination (R^2) of the targets.

@param test_targets : a Numpy array that represents the actual target values
@param predicted_targets : a Numpy array that represents the predicted target values

@returns the R^2 double/float value
"""

def get_and_print_R2_for_model(test_targets, predicted_targets):
    r2 = sklearn.metrics.r2_score(y_true=test_targets, y_pred=predicted_targets)
    print("R2 (Sklearn): " + str(r2))
  
    return r2



"""
This function uses the mean absolute percentage error (MAPE) to get a percentage error for a regression model. 

Note that if the output is a number above 1 or below 0, usually very big in magnitude, it is because 
some of the numbers in the test_targets are 0 or very close to 0, which gives a division by 0.

@param test_targets : a Numpy array that represents the actual target values
@param predicted_targets : a Numpy array that represents the predicted target values

@returns the MAPE double/float value which represents the decimal form of the percentage, so generally between 0 and 1
"""

def get_and_print_mape_for_model(test_targets, predicted_targets):
  mape = sklearn.metrics.mean_absolute_percentage_error(test_targets, predicted_targets)
  print("MAPE: " + str(mape))

  return mape



"""
This function gets the mean absolute error (MAE) for a regression model. 

Note that if the output is a number below 0, usually very big in magnitude, it is because 
some of the numbers in the test_targets are 0 or very close to 0, which gives a division by 0.

@param test_targets : a Numpy array that represents the actual target values
@param predicted_targets : a Numpy array that represents the predicted target values

@returns the MAPE double/float value, which should be greater than or equal to 0
"""
def get_and_print_mae_for_model(test_targets, predicted_targets):
  mae = sklearn.metrics.mean_absolute_error(test_targets, predicted_targets)
  print("MAE: " + str(mae))

  return mae



"""
This functions plots the actual and predicted values as a line plot.

@param indices : the x-values of the plot as a 1-column Numpy array
@param test_targets : a 1-column Numpy array that represents the actual target values
@param predicted_targets : a 1-column Numpy array that represents the predicted target values
"""
def plot_actual_and_predictions_line(x_val, test_targets, predicted_targets):
  # fig = plt.figure(figsize=(12, 8))
  # fig2 = plt.figure(figsize=(12, 8))
  
  # HR
  fig = plt.figure(figsize=(12, 8))
  plt.plot(x_val, test_targets[:, 0], label="Actual HR", c="b", linestyle="dashed")
  plt.plot(x_val, predicted_targets[:, 0], label="Predicted HR", c="lightblue", linestyle="dashed")
  plt.ylabel("HR Values")
  plt.title("Plot of Actual and Predicted Values - HR")
  plt.legend()

  # SPO2
  fig2 = plt.figure(figsize=(12, 8))
  plt.plot(x_val, test_targets[:, 1], label="Actual SPO2", c="purple")
  plt.plot(x_val, predicted_targets[:, 1], label="Predicted SPO2", c="lavender")
  plt.ylabel("SPO2 Values")
  plt.title("Plot of Actual and Predicted Values - SPO2")
  plt.legend()

  # NBP
  fig3 = plt.figure(figsize=(12, 8))
  plt.plot(x_val, test_targets[:, 2], label="Actual NBP", c="g")
  plt.plot(x_val, predicted_targets[:, 2], label="Predicted NBP", c="lightgreen")
  plt.ylabel("NBP Values")
  plt.title("Plot of Actual and Predicted Values - NBP")
  plt.legend()

  # ECG
  fig4 = plt.figure(figsize=(12, 8))
  plt.plot(x_val, test_targets[:, 3], label="Actual ECG", c="r")
  plt.plot(x_val, predicted_targets[:, 3], label="Predicted ECG", c="pink")
  plt.ylabel("ECG Values")
  plt.title("Plot of Actual and Predicted Values - ECG")
  plt.legend()

  # plt.ylabel("Values")
  # plt.title("Plot of Actual and Predicted Values")
  # plt.legend()
  plt.show()



"""
This functions plots the actual and predicted values as a scatter plot.

@param indices : a 1-column Numpy array that represents x-values of the plot 
@param test_targets : a 1-column Numpy array that represents the actual target values
@param predicted_targets : a 1-column Numpy array that represents the predicted target values
"""
def plot_actual_and_predictions_scatter(x_val, test_targets, predicted_targets):
  fig = plt.figure(figsize=(12, 8))
  plt.scatter(x_val, test_targets, label="Actual Values", c="b")
  plt.scatter(x_val, predicted_targets, label="Predicted Values", c="r")

  plt.ylabel("Value")
  plt.title("Plot of Actual and Predicted Values")
  plt.legend()




# Function for making LSTM model with a certain amount of layers
"""
This function makes a custom LSTM neural network model.

@param train_features : a Numpy array that represents the training data features
@param train_labels : a Numpy array that represents the training data targets
@param numOfUnits : a list that contains the number of units for each layer, which could be customized by providing different integers in the desired order. 
                    The size of the list should either be equal to 1 or numOfLayers. If size = 1, then all layers will have that number of units in the list.
                    (eg. [4, 8, 16] if numOfLayers = 3, or [4] for any value of numOfLayers)
@param numOfLayers : an integer that represents the desired of layers to add to the model 
@param epochNum : the number of iterations/epochs for running the model
@param learning_rate : a double/float that represents the learning rate of the model

@returns the LSTM neural network TensorFlow model and the result from fitting the model

"""
def make_custom_LSTM(train_features, train_labels, numOfUnits=[4], numOfLayers=3, batchNum=None, epochNum=100, learning_rate=0.01, min_delta=0.0007):
  callback_valLoss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, min_delta=min_delta, mode='min', verbose=1)
  callback_trainLoss = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=min_delta, mode='min', verbose=1)
  
  limit = numOfLayers

  if (len(numOfUnits) is not numOfLayers) and (len(numOfUnits) is not 1):
    raise Exception("Invalid input for numOfUnits.")
  
  for i in range(0, limit):
    if len(numOfUnits) is numOfLayers:
      numUnits = numOfUnits[i]
    
    else:
      numUnits = numOfUnits[0]

  lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.LSTM(units=numUnits, return_sequences=True),
    tf.keras.layers.Dense(1), #for 1 value output - 4 does not work
  ])

  lstm_model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                    loss='mean_squared_error', metrics=['accuracy'])
  
  lstm_model_fit = lstm_model.fit(train_features, train_labels, batch_size=batchNum, epochs=epochNum, validation_split=0.3, callbacks=[callback_valLoss, callback_trainLoss])

  return lstm_model, lstm_model_fit




"""
This function splits the training and test sets and turns a certain number of previous values as "features" and the next value as the "target"

@param train_array : a Numpy array with at least 2 columns of the training set values
@param test_array : a Numpy array with at least 2 columns of the test set values
@param numOfPrevValues : an integer that represents the number of previous rows of values to use as features
@param targetIndex : an integer that represents the column number to use for target values

@returns four Numpy arrays for the previous values in the training set, the next 1 value in the training set, the previous values
         in the test set, and the next 1 value in the test set
"""
def splitDataToFeaturesTargetValues(train_array, test_array, numOfPrevValues=3, targetIndex=1):

  # prevData - features
  # nextData - labels

  prevData_train = []
  nextData_train = [] 
  prevData_test = []
  nextData_test = []

  for i in range(numOfPrevValues, len(train_array)):
    prevData_train.append(train_array[i-numOfPrevValues: i]) # append features as it's appending alls rows
    nextData_train.append(train_array[i]) # append label as it's only the row at the target colun

  # don't want to split test data this way
  for j in range(numOfPrevValues, len(test_array)):
    prevData_test.append(test_array[j-numOfPrevValues: j])
    nextData_test.append(test_array[j])  

  prevData_train = np.array(prevData_train)
  nextData_train = np.array(nextData_train).reshape(len(nextData_train), 4)
  print("TRAIN")
  print(f"This is shape of train features array: {prevData_train.shape}")
  print(f"This is shape of train labels array: {nextData_train.shape}")

  prevData_test = np.array(prevData_test)
  nextData_test = np.array(nextData_test).reshape(len(nextData_test), 4) # used to be 1d array w many columns but one row, now it's many rows 1 column so it's a 2d array i think
  print("TEST")
  print(f"This is shape of test features array: {prevData_test.shape}")
  print(f"This is shape of test labels array: {nextData_test.shape}")

  return prevData_train, nextData_train, prevData_test, nextData_test


# In[ ]:


# Scaling and splitting dataset, using R-Wave Voltages and SpO2 as features
# pd.set_option("max_rows", 1000)
print(case4_filteredData.head(20))

case4_filteredData_arr_scaled, minMaxScaler, minMaxScaleFactor, case4_filteredData_min, case4_filteredData_max = minMaxScaling(case4_filteredData)
print("case 4 max: "), print(case4_filteredData_max)
print("case 4 min: "), print(case4_filteredData_min)

# Add more columns names if more health variables are added.
case4_filteredData_scaled = pd.DataFrame(case4_filteredData_arr_scaled, columns=["HR", "SpO2","NBP (Mean)","ECG"])

# print()
# print(case4_filteredData_scaled[0:20])

listOfVariables = ["HR", "SpO2","NBP (Mean)","ECG"] # Add more columns names if more health variables are added.
listOfVariables_len = len(listOfVariables) - 1 

data_train, data_test = split_data_train_test(np.array(case4_filteredData_scaled[listOfVariables]), train_size=0.8) # scaled data

print("\nOriginal data train: ")
print(data_train[0:20])

print(" Original test")
print(data_test[0:20])

print(" Here to see the size of each data: ")
print("This is the size of the train data: ")
print(data_train.shape)
print("This is the size of the test data: ")
print(data_test.shape)


# # General functions

# # LSTM Model - Predicting HR

# In[ ]:


# Running the models to predict all 4 variables

data_train_features, data_train_label, data_test_features, data_test_label = splitDataToFeaturesTargetValues(data_train, data_test, numOfPrevValues=4 ) #numofPrevValues only works with 4
print(data_train_features.shape)

#batch_num = int(len(data_train_label) / 1550)
batch_num = None
epochNum = 1000
learn_rate = 0.0001
numOfUnits_list = [512]

print("\nBatch Size: " + str(batch_num) + "\n")

lstm_model, lstm_model_fit = make_custom_LSTM(data_train_features, data_train_label, numOfUnits=numOfUnits_list, batchNum=batch_num,
                                              numOfLayers=3, epochNum=epochNum, learning_rate=learn_rate, min_delta=0.0001)


plot_loss(lstm_model_fit, 0, epochNum)
lstm_model.summary()


# In[ ]:


# Predicting with the model - training set - LSTM

# making the predictions  -------------------------------------------------------------------------------------
print("\nFor training set:\n")
predictions = lstm_model.predict(data_train_features)
print("Shape of predictions BEFORE reshape: "), print(predictions.shape)
print(predictions)

# reshape into 2d array? into the shape of the train features which is 3d
predictions = np.reshape(predictions,(41908, 4))
#predictions = predictions.reshape(data_train_features.shape[0], data_train_features.shape[1]) # current shape is (length, numOfPrevValues, 1) - new shape (length, numOfPrevValues, 4)
#predictions = predictions[:, 0] # getting values for the next time step (t) and not (t+1, t+2, or t+3) 
#predictions = np.reshape(predictions, newshape=(len(predictions), 1))
print("Shape of predictions AFTER reshape: "), print(predictions.shape)

# plotting accuracies and losses -------------------------------------------------------------------------------
get_and_print_rmse_for_model(data_train_label, predictions)
get_and_print_R2_for_model(data_train_label, predictions)
get_and_print_mae_for_model(data_train_label, predictions)
get_and_print_mape_for_model(data_train_label, predictions)

plot_actual_and_predictions_line(np.arange(0, len(data_train_label)), data_train_label, predictions)
# plot_actual_and_predictions_scatter(np.arange(0, len(data_train_label)), data_train_label, predictions)

print("\nAfter reversing the scaling:\n")

# For if MinMaxScaler was used; change the column number/index to get the values for other health variables
predictions_reversed = ((predictions - 0.1) / (1 - 0.1)) * (case4_filteredData_max[3] - case4_filteredData_min[3]) + case4_filteredData_min[3]
data_train_label_reversed = ((data_train_label - 0.1) / (1 - 0.1)) * (case4_filteredData_max[3] - case4_filteredData_min[3]) + case4_filteredData_min[3]
# data_train_features_reversed = ((data_train_features - 0.1) / (1 - 0.1)) * (combinedData_max[0] - combinedData_min[0]) + combinedData_min[0]

# For if standard scaling was used; change the column number/index to get the values for other health variables
# predictions_reversed = predictions * combinedData_stdDev[1] + combinedData_mean[1]
# data_train_label_reversed = data_train_label * combinedData_stdDev[1] + combinedData_mean[1]
# data_train_features_reversed = data_train_features * combinedData_stdDev[0] + combinedData_mean[0]

get_and_print_rmse_for_model(data_train_label_reversed, predictions_reversed)
get_and_print_R2_for_model(data_train_label_reversed, predictions_reversed)
get_and_print_mae_for_model(data_train_label_reversed, predictions_reversed)
get_and_print_mape_for_model(data_train_label_reversed, predictions_reversed)


# In[ ]:


# Predicting with the model- test set - LSTM---------------- NEW METHOD

print("\nFor testing set:\n")

# set initial values
numOfPrevValues = 4
batch = []
all_values = []

# ALL IMPLEMENTED INTO THE LOOP
for i in range(0, len(data_test_features) - numOfPrevValues):
  print(i)
  # 1. set the batch accordingly
  if i == 0:
    # only first batch comes from data_test features
    batch = data_test_features[0]
    all_values = batch
  else:
    batch = all_values[i:]

  # 2. reshape the batch
  # old shape (2d) - (27,4)
  # new shape (3d) - (1,27,4) - need to be 3d because shape of model input is also 3d 
  batch = batch.reshape(1, data_test_features[0].shape[0], data_test_features[0].shape[1])

  # 3. make the prediction
  predicted_val = lstm_model.predict(batch)
  # revert reshaping to 2d
  batch = batch.reshape(data_test_features[0].shape[0], data_test_features[0].shape[1])

  # 4. append to all values array
  # all_values = batch
  print(predicted_val.shape)
  print(all_values.shape)
  predicted_val = np.reshape(predicted_val,(1,4))
  all_values = np.append(all_values, [predicted_val[0]], axis=0)
  all_values = np.array(all_values)

  # 5. print everything
  print(f"LOOP #{i}:")
  print(f"Batch {i} shape: {batch.shape}")
  # print(batch)
  print(f"Predicted {i} shape: {predicted_val.shape}")
  # print(predicted_val)
  print(f"All predicted values shape: {all_values.shape}")
  # print(all_values)


# In[ ]:


# Predicting with the model- test set - LSTM

# print("\nFor testing set:\n")

# predictions = ann_model_hr.predict(data_test_features)

# predictions = predictions.reshape(data_test_features.shape[0], data_test_features.shape[1]) # current shape is (length, numOfPrevValues, 1)
# predictions = predictions[:, 0] # getting values for the next time step (t) and not (t+1, t+2, or t+3)
# predictions = np.reshape(predictions, newshape=(len(predictions), 1))
# predictions[predictions > 1.0] = 1.0

print(data_test_label.shape)
get_and_print_rmse_for_model(data_test_label, all_values)
get_and_print_R2_for_model(data_test_label, all_values)
get_and_print_mae_for_model(data_test_label, all_values)
get_and_print_mape_for_model(data_test_label, all_values)

plot_actual_and_predictions_line(np.arange(0, len(data_test_label)), data_test_label, all_values)
# plot_actual_and_predictions_scatter(np.arange(0, len(data_test_label)), data_test_label, all_values)

# # this function won't plot properly if there are more than 3 variables in the model
# plot_model_data3D(data_test_features[:, :, 0][:, 0].flatten(), data_test_features[:, :, 2][:, 0].flatten(), data_test_features[:, :, 1][:, 0].flatten(),
#                  predictions.flatten(), "R-Wave Voltages", "SpO2", "HR") 

# column 1 for HR values, 0 for ECG R-Wave Voltages
print("\nAfter reversing the scaling: \n")

# For if MinMaxScaler was used; change the column number/index to get the values for other health variables
predictions_reversed = ((all_values - 0.1) / (1 - 0.1)) * (case4_filteredData_max[3] - case4_filteredData_max[3]) + case4_filteredData_max[3]
data_test_label_reversed = ((data_test_label - 0.1) / (1 - 0.1)) * (case4_filteredData_max[3] - case4_filteredData_max[3]) + case4_filteredData_max[3]
# data_test_features_reversed = ((data_test_features - 0.1) / (1 - 0.1)) * (combinedData_max[0] - combinedData_min[0]) + combinedData_min[0]
# predictions_reversed[predictions_reversed > 100] = 100

# For if standard scaling was used; change the column number/index to get the values for other health variables
# predictions_reversed = predictions * combinedData_stdDev[1] + combinedData_mean[1]
# data_test_label_reversed = data_test_label * combinedData_stdDev[1] + combinedData_mean[1]
# data_test_features_reversed = data_test_features * combinedData_stdDev[0] + combinedData_mean[0]

get_and_print_rmse_for_model(data_test_label_reversed, predictions_reversed)
get_and_print_R2_for_model(data_test_label_reversed, predictions_reversed)
get_and_print_mae_for_model(data_test_label_reversed, predictions_reversed)
get_and_print_mape_for_model(data_test_label_reversed, predictions_reversed)


# In[ ]:


#Plotting results

#HR
fig = plt.figure(figsize=(12, 8))
xtrain_set = np.arange(0, len(data_train_features))
ytrain_set = np.array(data_train_label[:,0])
plt.plot(xtrain_set, ytrain_set, 'b', label="Actual Train")

xtest_set = np.arange(len(data_train_features),len(data_train_features) + len(data_test_features))
ytest_set = np.array(data_test_label[:,0])
plt.plot(xtest_set, ytest_set, 'r', label="Actual Test")

pred_train_sig = np.array(predictions[:,0])
plt.plot(xtrain_set, pred_train_sig, 'orange', label="Predicted Train", linestyle="dotted")

pred_test_sig = np.array(all_values[:,0])
plt.plot(xtest_set, pred_test_sig,'green', label="Predicted Test", linestyle="dotted")
plt.title("Heart Rate (HR)")
plt.legend()
plt.show()



#SpO2
fig = plt.figure(figsize=(12, 8))
xtrain_set = np.arange(0, len(data_train_features))
ytrain_set = np.array(data_train_label[:,1])
plt.plot(xtrain_set, ytrain_set, 'b', label="Actual Train")

xtest_set = np.arange(len(data_train_features),len(data_train_features) + len(data_test_features))
ytest_set = np.array(data_test_label[:,1])
plt.plot(xtest_set, ytest_set, 'r', label="Actual Test")

pred_train_sig = np.array(predictions[:,1])
plt.plot(xtrain_set, pred_train_sig, 'orange', label="Predicted Train", linestyle="dotted")

pred_test_sig = np.array(all_values[:,1])
plt.plot(xtest_set, pred_test_sig,'green', label="Predicted Test", linestyle="dotted")
plt.title("SpO2")
plt.legend()
plt.show()





#NBP
fig = plt.figure(figsize=(12, 8))
xtrain_set = np.arange(0, len(data_train_features))
ytrain_set = np.array(data_train_label[:,2])
plt.plot(xtrain_set, ytrain_set, 'b', label="Actual Train")

xtest_set = np.arange(len(data_train_features),len(data_train_features) + len(data_test_features))
ytest_set = np.array(data_test_label[:,2])
plt.plot(xtest_set, ytest_set, 'r', label="Actual Test")

pred_train_sig = np.array(predictions[:,2])
plt.plot(xtrain_set, pred_train_sig, 'orange', label="Predicted Train", linestyle="dotted")

pred_test_sig = np.array(all_values[:,2])
plt.plot(xtest_set, pred_test_sig,'green', label="Predicted Test", linestyle="dotted")
plt.title("NBP")
plt.legend()
plt.show()





#ECG
fig = plt.figure(figsize=(12, 8))
xtrain_set = np.arange(0, len(data_train_features))
ytrain_set = np.array(data_train_label[:,3])
plt.plot(xtrain_set, ytrain_set, 'b', label="Actual Train")

xtest_set = np.arange(len(data_train_features),len(data_train_features) + len(data_test_features))
ytest_set = np.array(data_test_label[:,3])
plt.plot(xtest_set, ytest_set, 'r', label="Actual Test")

pred_train_sig = np.array(predictions[:,3])
plt.plot(xtrain_set, pred_train_sig, 'orange', label="Predicted Train", linestyle="dotted")

pred_test_sig = np.array(all_values[:,3])
plt.plot(xtest_set, pred_test_sig,'green', label="Predicted Test", linestyle="dotted")
plt.title("ECG")
plt.legend()
plt.show()

