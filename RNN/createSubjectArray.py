import numpy as np
import math
import torch
import dgl
from dgl.data.utils import save_graphs
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import statistics

def createSubjArr(df, columns=None):
    if columns is None:
        columns = df.columns
    subjects = dict()
    labels = dict()
    for index, row in df.iterrows():
        if row["RSUBJID"] not in subjects.keys():
            subjects[row["RSUBJID"]] = list()
            labels[row["RSUBJID"]] = list()
        
        entry = [row[column] for column in columns]
        print(entry)
        print(type(entry))
        subjects[row["RSUBJID"]].append(entry)
        labels[row["RSUBJID"]].append(row["PostCond"])
    
    subject_list = list()
    label_list = list()
    for key in subjects.keys():
        print(subjects[key])
        print(type(subjects[key]))
        subject_list.append(torch.tensor(subjects[key]).float())
        label_list.append(statistics.mode(labels[key]))

    return subject_list, np.array(label_list)