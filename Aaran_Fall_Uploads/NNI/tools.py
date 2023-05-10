import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from xgboost import XGBRegressor

def create_subj_dict(df):
    subjects = dict()
    for index, row in df.iterrows():
        if row["RSUBJID"] not in subjects.keys():
            subjects[row["RSUBJID"]] = list()

        subjects[row["RSUBJID"]].append(row.drop(columns="RSUBJID"))
    return subjects

def find_threshold_num_visits(subj_dict, threshold, percentage=True):
    num_visit_count = dict()
    for k in subj_dict.keys():
        instances = len(subj_dict[k])
        if instances not in num_visit_count.keys():
            num_visit_count[instances] = 0
        num_visit_count[instances] += 1
    
    num_subj = len(subj_dict)
    if percentage:
        thresh = num_subj * (1 - threshold)
    else:
        thresh = num_subj - threshold
    running_sum = 0
    sorted_keys = list(num_visit_count.keys())
    sorted_keys.sort()
    for k in sorted_keys:
        running_sum += num_visit_count[k]
        # print(running_sum)
        # print(k)
        if running_sum >= thresh:
            return k

# requires that entries for each subject in subj_dict are entered in chronological order
# min_instances only has effect if custom_filterer is None
def avg_interval(subj_dict, min_instances=2, per_instance=True, custom_filterer=None):
    def filterer(pair):
        key, value = pair
        if len(value) >= min_instances:
            return True
        return False
    if custom_filterer is None:
        multi_subj = dict(filter(filterer, subj_dict.items()))
    else:
        multi_subj = dict(filter(custom_filterer, subj_dict.items()))
    sum = 0
    for k in multi_subj.keys():
        interval = multi_subj[k][-1]["RDAYSFROMINDEX"] - multi_subj[k][0]["RDAYSFROMINDEX"]
        if per_instance:
            interval = interval / (len(multi_subj[k]) - 1)
        sum += interval
    print(len(multi_subj))
    if len(multi_subj) == 0:
        return None
    return (sum / len(multi_subj))

def graph_col(subj_dict, subj_id, col_name):
    def one_col(row):
        return row[col_name]
    def index_col(row):
        return row["RDAYSFROMINDEX"]
    plt.scatter(list(map(index_col, subj_dict[subj_id])), list(map(one_col, subj_dict[subj_id])))

def subj_split(subj, interval):
    split_subj = dict()
    sections = 1
    split_subj[sections] = list()
    first_ind_of_section = 0
    for i in range(len(subj)):
        if subj[i]["RDAYSFROMINDEX"] - first_ind_of_section > interval:
            sections += 1
            split_subj[sections] = list()
            first_ind_of_section = subj[i]["RDAYSFROMINDEX"]
        split_subj[sections].append(subj[i])
    return split_subj
        



def split_dict(subj_dict, interval, filterer=None):
    new_dict = dict()
    if filterer is None:
        filtered_dict = subj_dict.copy()
    else:
        filtered_dict = dict(filter(filterer, subj_dict))
    for k in filtered_dict.keys():
        new_dict[k] = subj_split(filtered_dict[k], interval)
    return new_dict

def flatten_split_dict(split_dict):
    new_dict = dict()
    for outer_key in split_dict.keys():
        for inner_key in split_dict[outer_key].keys():
            new_key = float(str(int(outer_key)) + "." + str(int(inner_key)))
            new_dict[new_key] = split_dict[outer_key][inner_key]
    return new_dict

def remove_missing(subj_dict, col):
    new_dict = dict()
    for k in subj_dict.keys():
        new_list = list()
        for row in subj_dict[k]:
            if not math.isnan(row[col]):
                new_list.append(row)
        if len(new_list) > 0:
            new_dict[k] = new_list
    return new_dict

def polyfit_subj(subj, col):
    irr_time_series = dict()
    for column in col:
        irr_time_series[column] = list()
    for entry in subj:
        for column in col:
            if not math.isnan(entry[column]):
                irr_time_series[column].append((entry["RDAYSFROMINDEX"], entry[column]))
    reg_timeseries = dict()
    for k in irr_time_series.keys():
        irr_time_series[k] = np.array(irr_time_series[k])
        xx = irr_time_series[k][:, 0] - irr_time_series[k][:, 0][0]
        yy = irr_time_series[k][:, 1]
        polynomial = np.polyfit(xx, yy, 3)
        reg_timeseries[k] = np.polyval(polynomial, list(range(0, 1100, 100)))
        print(f'Length of regular time series for {k}: {len(reg_timeseries[k])}')
    return reg_timeseries

def resample_dict(subj_dict, col):
    time_dict = dict()
    for k in subj_dict.keys():
        time_dict[k] = polyfit_subj(subj_dict[k], col)
    return time_dict

def graph_col_with_fit(subj_dict, id, col):
    graph_col(subj_dict, id, col)
    plt.plot(list(range(int(subj_dict[id][0]["RDAYSFROMINDEX"]), int(subj_dict[id][0]["RDAYSFROMINDEX"]) + 1100, 100)), polyfit_subj(subj_dict[id], [col])[col])

def return_trends(df, age_groups, cols):
    trends = dict()
    for col in cols:
        trends[col] = dict()
    for ages in age_groups:
        for index, row in df.iterrows():
            # print(row)
            if row["AGE_G"] in ages:
                # print("here")
                for col in cols:
                    if not np.isnan(row[col]):
                        if ages[0] not in trends[col].keys():
                            trends[col][ages[0]] = list()
                        trends[col][ages[0]].append(row[col])
    for col_k in trends:
        for age_k in trends[col_k]:
            trends[col_k][age_k] = np.sum(trends[col_k][age_k]) / len(trends[col_k][age_k])
    return trends

def return_trends_from_dict(subj_dict, age_groups, cols):
    trends = dict()
    for col in cols:
        trends[col] = dict()
    for ages in age_groups:
        for subj in subj_dict.keys():
            for row in subj_dict[subj]:
                # print(row)
                if row["AGE_G"] in ages:
                    # print("here")
                    for col in cols:
                        if not np.isnan(row[col]):
                            if ages[0] not in trends[col].keys():
                                trends[col][ages[0]] = list()
                            trends[col][ages[0]].append(row[col])
    for col_k in trends:
        for age_k in trends[col_k]:
            trends[col_k][age_k] = np.sum(trends[col_k][age_k]) / len(trends[col_k][age_k])
    return trends

def drop_bad_rows(subj_dict, thresh):
    new_dict = dict()
    for k in subj_dict.keys():
        count = 0
        for row in subj_dict[k]:
            count = 0
            for index, col in row.items():
                if np.isnan(col):
                    count += 1
            if count < thresh:
                if k not in new_dict.keys():
                    new_dict[k] = list()
                new_dict[k].append(row)
    return new_dict

def dist_from_age_mean(subj_dict, trends):
    new_dict = dict()
    for k in subj_dict.keys():
        for row in subj_dict[k]:
            if row["AGE_G"] < min(list(trends[list(trends.keys())[0]].keys())) or row["AGE_G"] > max(list(trends[list(trends.keys())[0]].keys())):
                continue
            if k not in new_dict.keys():
                new_dict[k] = list()
            new_row = dict()
            for row_k in row.keys():
                if row_k in trends.keys() and row["AGE_G"] :
                    new_row[row_k] = row[row_k] - trends[row_k][row["AGE_G"]]
                else:
                    new_row[row_k] = row[row_k]
            new_dict[k].append(new_row)
    return new_dict

def dict_to_linfit(subj_dict, trend_cols):
    new_dict = dict()
    for subj in subj_dict.keys():
        new_dict[subj] = dict()
        subj_trends = dict()
        for row in subj_dict[subj]:
            for col in row.keys():
                if col not in trend_cols and col != "AGE_G":
                    if col not in new_dict[subj].keys():
                        new_dict[subj][col] = row[col]
                else:
                    if col not in subj_trends.keys():
                        subj_trends[col] = list()
                    subj_trends[col].append((row["AGE_G"], row["RDAYSFROMINDEX"], row[col]))
        for col in subj_trends.keys():
            init_age = 15 * 365 + subj_trends[col][0][0] *  5 * 365
            last_age = init_age + subj_trends[col][-1][1]
            if "start_age" not in new_dict[subj].keys():
                new_dict[subj]["start_age"] = init_age
                new_dict[subj]["end_age"] = last_age
            xx = np.array([init_age + val[1] for val in subj_trends[col]])
            yy = np.array([val[2] for val in subj_trends[col]])
            coeffs = np.polyfit(x = xx, y = yy, deg=1)
            # print(f'Number of coeffs returned: {len(coeffs)}')
            for i in range(len(coeffs)):
                col_name = str(col) + "_" + str(i)
                new_dict[subj][col_name] = coeffs[i]
            new_dict[subj][str(col) + "_" + str("interval")] = last_age * coeffs[0] - init_age * coeffs[0]
            # new_dict[subj][str(col) + "_" + str("start")] = last_age * coeffs[0] + coeffs[1]
    return new_dict

def subj_dict_to_df(subj_dict):
    ret_list = list()
    for subj in subj_dict.keys():
        for row in subj_dict[subj]:
            new_row = dict()
            new_row["RSUBJID"] = subj
            for col in row.keys():
                new_row[col] = row[col]
            ret_list.append(new_row)
    return pd.DataFrame(ret_list)

def lin_dict_to_df(lin_dict):
    ret_list = list()
    col_names = list()
    for subj in lin_dict.keys():
        new_row = dict()
        new_row["RSUBJID"] = subj
        local_cols = list()
        for col in lin_dict[subj].keys():
            local_cols.append(col)
            if col not in col_names:
                col_names.append(col)
            new_row[col] = lin_dict[subj][col]
        if local_cols != col_names:
            print(f'Patient {subj} does not have columns {[column for column in col_names if column not in local_cols]}')
        ret_list.append(new_row)
    return pd.DataFrame(ret_list)

def subj_dict_nan(subj_dict):
    has_nan = False
    for lst in subj_dict.values():
        for d in lst:
            for val in d.values():
                if np.isnan(val):
                    has_nan = True
                    break
            if has_nan:
                break
        if has_nan:
            break
    return has_nan

def lin_dict_nan(lin_dict):
    has_nan = False
    for dic in lin_dict.values():
        for val in dic.values():
            if np.isnan(val):
                has_nan = True
                break
        if has_nan:
            break
    return has_nan

def check_subj_dict_dim_err(subj_dict):
    num_cols = len(subj_dict[list(subj_dict.keys())[0]][0])
    dim_err = False
    has_empty_subj = False
    for subj in subj_dict.keys():
        if len(subj_dict[subj]) == 0:
            print(f'subject {subj} has no information')
            has_empty_subj = True
        for row in subj_dict[subj]:
            if len(row) != num_cols:
                dim_err = True
                break
        if dim_err:
            break
    return has_empty_subj or dim_err


def check_lin_dict_dim_err(lin_dict):
    num_cols = len(lin_dict[list(lin_dict.keys())[0]])
    dim_err = False
    for subj in lin_dict.keys():
        if len(lin_dict[subj]) != num_cols:
            dim_err = True
            break
    return dim_err

def xg_imputer_fit(df, train_cols, drop_train_nan = True):
    new_var_models = dict()
    if drop_train_nan:
        df = df.dropna(axis=0, subset=train_cols)
        train_df = df[train_cols]
    new_var_models = dict()
    for col in df.drop(labels=train_cols, axis=1).columns:
        new_var_models[col] = XGBRegressor()
        frame = pd.concat([train_df, df[[col]]], axis=1)
        frame = frame.dropna(axis=0)
        new_var_models[col].fit(frame[train_cols], frame[col])
    return new_var_models

def xg_imputer_transform(df, new_var_models, input_vars):
    df_input = df[input_vars]
    new_df = df.copy(deep=True)
    for col in new_var_models.keys():
        pred = new_var_models[col].predict(df_input)
        new_df[col] = pred
    return new_df