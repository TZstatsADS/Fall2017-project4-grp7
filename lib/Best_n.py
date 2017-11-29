#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:24:39 2017

@author: Chloe
"""
import numpy as np
import pandas as pd
import pickle


def main():
###############################data1
    train1 = pd.read_csv('./output/train1_df.csv',index_col = 0)


    with open('./PearsonCosineMatrix/pearson_correlation1.pkl', 'rb') as f1:
        pearson_corr1 = pickle.load(f1)
    pearson_corr1 = pearson_corr1.values

###############################data2
    train2 = pd.read_csv('./output/train2_df.csv',index_col = 0)


    with open('./PearsonCosineMatrix/pearson_correlation2.pkl', 'rb') as f2:
        pearson_corr2 = pickle.load(f2)
    pearson_corr2 = pearson_corr2.values

    weights_neighbor1, neighbors1 = selecting_neighborhood(20, pearson_corr1, train1)
    weights_neighbor2, neighbors2 = selecting_neighborhood(20, pearson_corr2, train2)
##########################Best-n estimator
def selecting_neighborhood(n, similarity_weights,data):
    data=data.values
    n_users = data.shape[0]
    weights_neighbor = np.zeros((n_users, n))
    neighbors = np.zeros((n_users, n))
    for a in range(n_users):
        arr = similarity_weights[a,:]
        weights_neighbor[a,:] = arr[np.argsort(arr)[-n:]]
        neighbors[a,:] = arr.argsort()[-n:][::-1]
    neighbors = neighbors.astype(int)
    return [weights_neighbor, neighbors]



