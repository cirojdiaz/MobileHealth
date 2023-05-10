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
import model_tools as tools

# original downsample function for ANN one to one with simulated HR data
# def downsample_data(window, values):
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

# dynamic downsample function


def load_data(input_file_name, colnames = ['TIME', 'HR'], usecols = ['HR']):
    #Need to allow code to access the CSV Dataset. 
    #Setting up an environment variable. Will add a readme file in order to show how to use this method.

    csv_folder_path = os.environ.get("CSV_Folder_Path")
    simulated_data_file_path = os.path.join(csv_folder_path, input_file_name)

    # FILTERING THE DATA
    # user1 = pd.read_csv('dataset/1.csv', names=colnames, header=None)
    df = pd.read_csv(simulated_data_file_path, names=colnames,header=None, index_col=False, error_bad_lines=False , usecols=usecols)


    df = df.dropna() #drop rows with NaN
    return df

def process_data(data, window, column_names, numOfPrevValues):
    new_data = tools.downsample_data(100, data, column_names)
    new_data_arr_scaled, minMaxScaler, minMaxScaleFactor, new_data_min, new_data_max = tools.minMaxScaling(new_data)
    new_data_scaled = pd.DataFrame(new_data_arr_scaled, columns=column_names)
    listOfVariables_len = len(column_names) - 1
    new_data_train, new_data_test = tools.split_data_train_test(np.array(new_data_scaled[column_names]), train_size=0.8) # scaled data
    data_train_features, data_train_label, data_test_features, data_test_label = tools.splitDataToFeaturesTargetValues(new_data_train, new_data_test, numOfPrevValues=numOfPrevValues)
    return data_train_features, data_train_label, data_test_features, data_test_label, minMaxScaler, minMaxScaleFactor, new_data_min, new_data_max

def train_model(data_train_features, data_train_label, data_test_features, data_test_label, batch_num = None, epoch_num = 5000, learn_rate = 0.000001, numOfUnits_list = [512], numOfPrevValues = 1000):
    ann_model_hr, ann_model_hr_fit = tools.make_custom_ANN(data_train_features, data_train_label, numOfUnits=numOfUnits_list, batchNum=batch_num,
                                           numOfLayers=3, epochNum=epoch_num, learning_rate=learn_rate, min_delta=0.0001)
    
    ann_model_hr.summary()
    return ann_model_hr, ann_model_hr_fit

def make_prediction(features, model):
    predictions = model.predict(features)
    return predictions

def get_results(predictions, labels, with_plot = False):
    rmse = tools.get_and_print_rmse_for_model(labels, predictions)
    r2 = tools.get_and_print_R2_for_model(labels, predictions)
    mae = tools.get_and_print_mae_for_model(labels, predictions)
    mape = tools.get_and_print_mape_for_model(labels, predictions)
    if with_plot:
        tools.plot_actual_and_predictions_line(np.arange(0, len(labels)), labels, predictions)
    return rmse, r2, mae, mape

def min_max_scaler_reverse(predictions, labels, max, min):
    predictions_reversed = ((predictions - 0.1) / (1 - 0.1)) * (max[0] - min[0]) + min[0]
    data_train_label_reversed = ((labels - 0.1) / (1 - 0.1)) * (max[0] - min[0]) + min[0]
    return predictions, labels

def long_term_prediction(test_features, model, numOfPrevValues):
    # set initial values
    batch = []
    all_values = []

    # ALL IMPLEMENTED INTO THE LOOP
    for i in range(0, len(test_features)):
        print(i)
        # 1. set the batch accordingly
        if i == 0:
            # only first batch comes from data_test features
            batch = test_features[0]
            all_values = batch
        else:
            batch = all_values[i:]

        # 2. reshape the batch
        # old shape (2d) - (27,4)
        # new shape (3d) - (1,27,4) - need to be 3d because shape of model input is also 3d 
        batch = batch.reshape(1, test_features[0].shape[0], test_features[0].shape[1])

        # 3. make the prediction
        predicted_val = model.predict(batch)
        # revert reshaping to 2d
        batch = batch.reshape(test_features[0].shape[0], test_features[0].shape[1])

        # 4. append to all values array
        # all_values = batch
        all_values = np.append(all_values, [predicted_val[0]], axis=0)
        all_values = np.array(all_values)

        # # 5. print everything
        # print(f"LOOP #{i}:")
        # print(f"Batch {i} shape: {batch.shape}")
        # # print(batch)
        # print(f"Predicted {i} shape: {predicted_val.shape}")
        # # print(predicted_val)
        # print(f"All predicted values shape: {all_values.shape}")
        # # print(all_values)
    if len(test_features[0]) != numOfPrevValues:
        raise Exception("feature size does not match numOfPrevValues")
    return all_values[numOfPrevValues:]

def test_model_from_df(data, colnames, downsample_window = 100, numOfPrevValues = 1000, batch_num = None, epoch_num = 5000, learn_rate = 0.000001, numOfUnits_list = [512]):
    data_train_features, data_train_labels, data_test_features, data_test_labels, minMaxScaler, minMaxScaleFactor, data_min, data_max = process_data(data, downsample_window, colnames, numOfPrevValues)
    model, model_fit = train_model(data_train_features, data_train_labels, data_test_features, data_test_labels, batch_num=batch_num, epoch_num=epoch_num, learn_rate=learn_rate, numOfUnits_list=numOfUnits_list, numOfPrevValues=numOfPrevValues)
    
    train_predictions = make_prediction(data_train_features, model)
    train_results = dict()
    train_results["rmse"], train_results["r2"], train_results["mae"], train_results["mape"] = get_results(data_train_labels, train_predictions)
    
    
    test_predictions = make_prediction(data_test_features, model)
    test_results = dict()
    test_results["rmse"], test_results["r2"], test_results["mae"], test_results["mape"] = get_results(data_test_labels, test_predictions)
    
    long_term_pred = long_term_prediction(data_test_features, model, numOfPrevValues)
    long_term_results = dict()
    long_term_results["rmse"], long_term_results["r2"], long_term_results["mae"], long_term_results["mape"] = get_results(data_test_labels, long_term_pred)

    return train_results, test_results, long_term_results, data_train_features, data_train_labels, data_test_features, data_test_labels, train_predictions, test_predictions, long_term_pred

def test_model_from_csv(input_file_name, colnames, usecols, downsample_window = 100, numOfPrevValues = 1000, batch_num = None, epoch_num = 5000, learn_rate = 0.000001, numOfUnits_list = [512]):
    data = load_data(input_file_name=input_file_name, colnames=colnames, usecols=usecols)
    train_results, test_results, long_term_results, data_train_features, data_train_labels, data_test_features, data_test_labels, train_predictions, test_predictions, long_term_prediction = test_model_from_df(data=data, colnames=colnames, downsample_window=downsample_window, numOfPrevValues=numOfPrevValues, batch_num=batch_num, epoch_num=epoch_num, learn_rate=learn_rate, numOfUnits_list=numOfUnits_list)
    return train_results, test_results, long_term_results, data_train_features, data_train_labels, data_test_features, data_test_labels, train_predictions, test_predictions, long_term_prediction





