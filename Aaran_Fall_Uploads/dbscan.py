import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN

def transform(X, y):
        dim = len(X[0])
        distances = find_nearest_neighbor(X, 2 * dim)
        knee = knee_locator(distances)
        # print(f'Knee: {knee}')
        # print(f'2 * dim: {2 * dim}')

        new_X, new_y = remove_outliers(X, y, knee, 2 * dim)
        return new_X, new_y


def remove_outliers(X, y, eps, min_samples):
    res = dbscan(X, eps, min_samples)
    labels = res.labels_
    new_X, new_y = list(), list()
    for i in range(0, len(labels)):
        if labels[i] != -1:
            new_X.append(X[i])
            new_y.append(y[i])
    new_X = np.asarray(new_X)
    new_y = np.asarray(new_y)
    return new_X, new_y



def dbscan( X, eps, min_samples):
    # print(f'eps: {eps}')
    # print(f'min_samples: {min_samples}')
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)

def knee_locator(distances, plot = False):
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    if plot:
        fig = plt.figure(figsize=(5, 5))
        knee.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
        print(distances[knee.knee])
 
    return distances[knee.knee]


def find_nearest_neighbor(X, num_neighbors, plot=False, start_plot=None, end_plot=None):
    nearest_neighbors = NearestNeighbors(n_neighbors=num_neighbors + 1)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:,num_neighbors], axis=0)
    if plot:
        fig = plt.figure(figsize=(5, 5))
        plt.plot(distances)
        plt.xlabel("Points")
        plt.ylabel("Distance")
        if start_plot is not None:
            plt.ylim(start_plot, end_plot)
    return distances

def plot_distance(df):
    df1 = df.sort_values(by=list(df.columns))
    df2 = pd.DataFrame(columns=["index", "distance"])
    for i in range(0, len(df1) - 1):
        dist = math.dist(df1.iloc[i], df1.iloc[i + 1])
        df2 = df2.append({'index': i, 'distance': dist}, ignore_index=True)
    df2 = df2.sort_values(by=['distance'])
    plt.scatter(df2['index'], df2['distance'])
    plt.show()