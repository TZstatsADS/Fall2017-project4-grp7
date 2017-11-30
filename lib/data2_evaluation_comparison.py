#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:06:07 2017

@author: Chloe
"""

import pickle
import pandas as pd
import numpy as np

##############load data########################

with open('/Users/zhuxiaoxia/Desktop/proj4/pearson_correlation2.pkl', 'rb') as f1:
    pearson_correlation2 = pickle.load(f1)
pearson_correlation2 = pearson_correlation2.values

with open('/Users/zhuxiaoxia/Desktop/proj4/cosine_correlation2.pkl', 'rb') as f1:
    cosine_correlation2 = pickle.load(f1)
cosine_correlation2 = cosine_correlation2.values

with open('/Users/zhuxiaoxia/Desktop/proj4/entropy_correlation2.pkl', 'rb') as f1:
    entropy_correlation2 = pd.read_pickle(f1)
entropy_correlation2 = entropy_correlation2.values

with open('/Users/zhuxiaoxia/Desktop/proj4/simrank_correlation.pkl', 'rb') as f1:
    simrank_correlation = pd.read_pickle(f1)
simrank_correlation = simrank_correlation.values


         
                             
########################dfm+roc######################
def neighbor_size_dfm_roc2(data_train,data_test_long,data_test_square,similarity_weights):
    i_range = np.arange(start = 1, stop = 10, step = 1)
    roc_list = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values) 
        pred_dfm  = Deviation_for_mean2(data_train,data_test_long,weights_neighbor,neighbors)
        pred_dfm = pd.DataFrame(pred_dfm)
        eva= roc_sensitivity(data_test_square, pred_dfm, goodness = 4)
        roc_list.append(eva)
        print(i)
        n_opt = roc_list.index(max(roc_list))+1
    return roc_list, n_opt

roc_list, n_opt = neighbor_size_dfm_roc2(train2,test_2,test2,pearson_correlation2)

roc_list_list = []
n_opt_list_roc = []
for similairty in pearson_correlation2, cosine_correlation2, entropy_correlation2,simrank_correlation:
    roc_list, n_opt = neighbor_size_dfm2(train2,test_2,test2,similairty)
    print("similarity done")
    roc_list_list.append(roc_list)
    n_opt_list_roc.append(n_opt)
    
pearson_corr2_roc = roc_list_list[0]
cosine_corr2_roc = roc_list_list[1]
entropy_corr2_roc = roc_list_list[2]

########################dfm+mae######################
def neighbor_size_dfm_mae2(data_train,data_test_long,data_test_square,similarity_weights):
    i_range = np.arange(start = 1, stop = 10, step = 1)
    mae_list = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values) 
        pred_dfm  = Deviation_for_mean2(data_train,data_test_long,weights_neighbor,neighbors)
        pred_dfm = pd.DataFrame(pred_dfm)
        eva = mean_absolute_error(data_test_square, pred_dfm)
        mae_list.append(eva)
        print(i)
        n_opt = mae_list.index(max(mae_list))+1
    return mae_list, n_opt

mae_list, n_opt = neighbor_size_dfm_mae2(train2,test_2,test2,pearson_correlation2)

mae_list_list = []
n_opt_list_mae = []
for similairty in pearson_correlation2, cosine_correlation2, entropy_correlation2,simrank_correlation:
    mae_list, n_opt = neighbor_size_dfm_mae2(train2,test_2,test2,similairty)
    print("similarity done")
    mae_list_list.append(mae_list)
    n_opt_list_mae.append(n_opt)
    
pearson_corr2_mae = mae_list_list[0]
cosine_corr2_mae = mae_list_list[1]
entropy_corr2_mae = mae_list_list[2]


##################zscore+roc#######################
def neighbor_size_zscore_roc2(data_train,data_test_long,data_test_square,similarity_weights):
    i_range = np.arange(start = 1, stop = 2, step = 1)
    roc_list2 = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values) 
        pred_z_score  = z_score2(data_train,data_test_long,weights_neighbor,neighbors)
        pred_z_score = pd.DataFrame(pred_z_score)
        eva = roc_sensitivity(data_test_square, pred_z_score, goodness = 4)
        roc_list.append(eva)
        print(i)
        n_opt = roc_list2.index(max(roc_list2))+1
    return roc_list2, n_opt

roc_list2, n_opt = neighbor_size_zscore_roc2(train2,test_2,test2,pearson_correlation2)

roc_list_list2 = []
n_opt_list_roc2 = []
for similairty in pearson_correlation2, cosine_correlation2, entropy_correlation2,simrank_correlation:
    roc_list2, n_opt = neighbor_size_zscore_roc2(train2,test_2,test2,similairty)
    print("similarity done")
    roc_list_list2.append(roc_list)
    n_opt_list_roc2.append(n_opt)
    
pearson_corr2_roc2 = roc_list_list2[0]
cosine_corr2_roc2 = roc_list_list2[1]
entropy_corr2_roc2 = roc_list_list2[2]


########################zscore+mae######################
def neighbor_size_dfm_mae2(data_train,data_test_long,data_test_square,similarity_weights):
    i_range = np.arange(start = 1, stop = 10, step = 1)
    mae_list = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values) 
        pred_dfm  = z_score2(data_train,data_test_long,weights_neighbor,neighbors)
        pred_dfm = pd.DataFrame(pred_dfm)
        eva = mean_absolute_error(data_test_square, pred_dfm)
        mae_list.append(eva)
        print(i)
        n_opt = mae_list.index(max(mae_list))+1
    return mae_list, n_opt

mae_list, n_opt = neighbor_size_dfm_mae2(train2,test_2,test2,pearson_correlation2)

mae_list_list2 = []
n_opt_list_mae2 = []
for similairty in pearson_correlation2, cosine_correlation2, entropy_correlation2,simrank_correlation:
    mae_list, n_opt = neighbor_size_dfm_mae2(train2,test_2,test2,similairty)
    print("similarity done")
    mae_list_list2.append(mae_list)
    n_opt_list_mae2.append(n_opt)
    
pearson_corr2_mae = mae_list_list2[0]
cosine_corr2_mae = mae_list_list2[1]
entropy_corr2_mae = mae_list_list2[2]





















