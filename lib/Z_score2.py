#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:27:15 2017

@author: zhuxiaoxia
"""
import numpy as np
import pandas as pd

#################movie data##############
train2 = pd.read_csv('./output/train2_df.csv',index_col = 0)
test_2 = pd.read_csv('./eachmovie_sample/data_test.csv',index_col = 0)

#########################Z-score
def z_score2(data_train,data_test,weights_neighbor,neighbors):
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
        p.append(np.mean(data_train.values[a,:]) + np.std(data_train.values[a,:])*np.sum((data_train.values[neighbors[a,:],i-1] - np.mean(data_train.values[neighbors[a,:],]))/np.std(data_train.values[neighbors[a,:],])*weights_neighbor[a,:])/np.sum(weights_neighbor[a,:]))
    return(p)

#pearson+best n = 20
pred_z2 = z_score2(train2,test_2,weights_neighbor2,neighbors2)

          