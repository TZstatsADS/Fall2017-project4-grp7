#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:23:18 2017

@author: Chloe
"""

import numpy as np
import pandas as pd
import pickle

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


###########################significance weighting 
#<50 common rated items, significance weight = n/50
#>50 common rated items, significance weight = 1
def significance_weighting(data):
    n_users = data.shape[0]
    co_rated = np.zeros((n_users, n_users))
    sig_weight = np.zeros((n_users, n_users))
    for i in range(n_users):
        for j in range(n_users):
             co_rated[i,j] = np.count_nonzero(data[i,:]*data[j,:])
             if co_rated[i,j]>=50:
                 sig_weight[i,j] = 1
             else:
                 sig_weight[i,j] = co_rated[i,j]/50
    return sig_weight

sig_weights1 = significance_weighting(train1.values)
devalued_weights1 = pearson_corr1*sig_weights1 
sig_weights2 = significance_weighting(train2.values)
devalued_weights2 = pearson_corr2*sig_weights2
 