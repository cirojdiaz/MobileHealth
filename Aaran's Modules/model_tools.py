# import libraries
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf

""" Functions """
# has to be generalized to be included in model_tools.py
# def downsample_data(window):
#   df_temp = pd.DataFrame(columns=['HR'])
#   HR_sum = 0
#   SPO_sum = 0
#   NBP_sum = 0
#   ECG_sum = 0

#   for i in range(1, len(values) - 1):
#     HR_sum += values[i, 0]
#     if (i + 1) % window == 0:
#       HR_sum = HR_sum/window
    
#       row = {'HR':HR_sum}
#       df_temp = df_temp.append(row, ignore_index = True)
#       HR_sum = 0
#       SPO_sum = 0
#       NBP_sum = 0
#       ECG_sum = 0
#   return df_temp

#generalized downsample function
def downsample_data(window, data, column_names):
  df_temp = pd.DataFrame(columns=column_names)
  sums = [0] * len(column_names)

  for i in range(1, len(data.values) - 1):
    row = dict()
    for j in range(len(column_names)):
        sums[j] += data[column_names[j]][i]
    if (i + 1) % window == 0:
        for j in range(len(column_names)):
            sums[j] = sums[j]/window
            row[column_names[j]] = sums[j]
            sums[j] = 0
        df_temp = df_temp.append(row, ignore_index = True)
  return df_temp

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

@returns the MAE double/float value, which should be greater than or equal to 0
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

  # # SPO2
  # fig2 = plt.figure(figsize=(12, 8))
  # plt.plot(x_val, test_targets[:, 1], label="Actual SPO2", c="purple")
  # plt.plot(x_val, predicted_targets[:, 1], label="Predicted SPO2", c="lavender")
  # plt.ylabel("SPO2 Values")
  # plt.title("Plot of Actual and Predicted Values - SPO2")
  # plt.legend()

  # # NBP
  # fig3 = plt.figure(figsize=(12, 8))
  # plt.plot(x_val, test_targets[:, 2], label="Actual NBP", c="g")
  # plt.plot(x_val, predicted_targets[:, 2], label="Predicted NBP", c="lightgreen")
  # plt.ylabel("NBP Values")
  # plt.title("Plot of Actual and Predicted Values - NBP")
  # plt.legend()

  # # ECG
  # fig4 = plt.figure(figsize=(12, 8))
  # plt.plot(x_val, test_targets[:, 3], label="Actual ECG", c="r")
  # plt.plot(x_val, predicted_targets[:, 3], label="Predicted ECG", c="pink")
  # plt.ylabel("ECG Values")
  # plt.title("Plot of Actual and Predicted Values - ECG")
  # plt.legend()

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



"""
This function makes a custom artificial neural network model.

@param train_features : a Numpy array that represents the training data features
@param train_labels : a Numpy array that represents the training data targets
@param numOfUnits : a list that contains the number of units for each layer, which could be customized by providing different integers in the desired order. 
                    The size of the list should either be equal to 1 or numOfLayers. If size = 1, then all layers will have that number of units in the list.
                    (eg. [4, 8, 16] if numOfLayers = 3, or [4] for any value of numOfLayers)
@param numOfLayers : an integer that represents the desired of layers to add to the model 
@param epochNum : the number of iterations/epochs for running the model
@param learning_rate : a double/float that represents the learning rate of the model

@returns the artificial neural network TensorFlow model and the result from fitting the model

"""
def make_custom_ANN(train_features, train_labels, numOfUnits=[4], numOfLayers=3, batchNum=None, epochNum=100, learning_rate=0.01, min_delta=0.0007):
  callback_valLoss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, min_delta=min_delta, mode='min', verbose=1)
  callback_trainLoss = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=min_delta, mode='min', verbose=1)

  # ann_model = tf.keras.Sequential()
  limit = numOfLayers

#changed is not -> !=
  if (len(numOfUnits) != numOfLayers) and (len(numOfUnits) != 1):
    raise Exception("Invalid input for numOfUnits.")
  
  for i in range(0, limit):
    if len(numOfUnits) is numOfLayers:
      numUnits = numOfUnits[i]
    
    else:
      numUnits = numOfUnits[0]
  
  ann_model = tf.keras.models.Sequential([
    tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Convolution1D(64, 3, activation='relu'), #add convolutional layer (filters, kernel size) 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(1) # default activation is linear 
    ])

  ann_model.compile(optimizer=tf.optimizers.legacy.Adam(learning_rate=learning_rate),
                    loss='mean_squared_error', metrics=['accuracy'])
  
  ann_model_fit = ann_model.fit(train_features, train_labels, batch_size=batchNum, epochs=epochNum, validation_split=0.3, callbacks=[callback_valLoss, callback_trainLoss])
  
  # must uncomment above line and return ann_model_fit as well below
  return ann_model, ann_model_fit


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
  nextData_train = np.array(nextData_train).reshape(len(nextData_train), 1)
  print("TRAIN")
  print(f"This is shape of train features array: {prevData_train.shape}")
  print(f"This is shape of train labels array: {nextData_train.shape}")

  prevData_test = np.array(prevData_test)
  nextData_test = np.array(nextData_test).reshape(len(nextData_test), 1) # used to be 1d array w many columns but one row, now it's many rows 1 column so it's a 2d array i think
  print("TEST")
  print(f"This is shape of test features array: {prevData_test.shape}")
  print(f"This is shape of test labels array: {nextData_test.shape}")

  return prevData_train, nextData_train, prevData_test, nextData_test