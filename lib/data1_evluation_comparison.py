#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:09:54 2017

@author: zhuxiaoxia
"""

def neighbor_size_dfm1(data_train,data_test,similarity_weights):
    import matplotlib.pyplot as plt
    i_range = np.arange(start = 1, stop = 10, step = 1)
    score_list_ = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values) 
        pred_dfm  = Deviation_for_mean1(data_train,data_test,weights_neighbor,neighbors)
        pred_dfm = pd.DataFrame(pred_dfm)
        eva= ranked_score(data_test, pred_dfm, d = 0.1, alpha = 5)
        score_list.append(eva)
        print(i)
        n_opt = score_list.index(max(score_list))+1
    return score_list, n_opt


def neighbor_size_zscore1(data_train,data_test,similarity_weights):
    import matplotlib.pyplot as plt
    i_range = np.arange(start = 1, stop = 10, step = 1)
    score_list = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values) 
        pred_zscore  = z_score1(data_train,data_test,weights_neighbor,neighbors)
        pred_zscore  = pd.DataFrame(pred_zscore)
        eva = ranked_score(data_test, pred_zscore, d = 0.1, alpha = 5)
        score_list.append(eva)
        print(i)
        n_opt = score_list.index(max(score_list))+1 
    return score_list, n_opt

##########################ranked score################################
score_list_list = []
n_opt_list = []
for similairty in pearson_correlation1, cosine_correlation1, entropy_correlation1:
    score_list, n_opt = neighbor_size_dfm1(train1,test1,similairty)
    print("similarity done")
    score_list_list.append(score_list)
    n_opt_list.append(n_opt)


score_list_list_z = []
n_opt_list_z = []
for similairty in pearson_correlation1, cosine_correlation1, entropy_correlation1:
    score_list, n_opt = neighbor_size_zscore1(train1,test1,similairty)
    print("similarity done")
    score_list_list_z.append(score_list)
    n_opt_list_z.append(n_opt)
    
    
######################data1, Deviation for Mean, n=13#########################
pearson_corr1_rank_score1 = score_list_list[0]
cosine_corr1_rank_score1 = score_list_list[1]
entropy_corr1_rank_score1 = score_list_list[2]

######################data1, z_score, n=13#########################
pearson_corr1_rank_score1 = score_list_list_z[0]
cosine_corr1_rank_score1 = score_list_list_z[1]
entropy_corr1_rank_score1 = score_list_list_z[2]

