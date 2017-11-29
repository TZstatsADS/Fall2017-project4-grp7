#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:25:37 2017

@author: Chloe
"""
import numpy as np
import pandas as pd

#################movie data##############
train2 = pd.read_csv('./output/train2_df.csv',index_col = 0)
test_2 = pd.read_csv('./eachmovie_sample/data_test.csv',index_col = 0)


#########################Deviation for mean
def Deviation_for_mean2(data_train,data_test,weights_neighbor,neighbors):
    user = data_test['User']
    user = np.array(user)   
    user_id = np.array(data_train.index)
    user_id = user_id.searchsorted(user)
    movie = data_test['Movie']
    movie = np.array(movie)   
    movie_id = data_train.columns.values.astype(int)
    movie_id = movie_id.searchsorted(movie)
    
    p = []
    for a,i in zip(user_id, movie_id):
        p.append(np.mean(data_train.values[a,:]) + np.sum((data_train.values[neighbors[a,:],i] - np.mean(data_train.values[neighbors[a,:],]))*weights_neighbor[a,:])/np.sum(weights_neighbor[a,:]))
    return p

pred_dfm2 = Deviation_for_mean2(train2,test_2,weights_neighbor2,neighbors2)