#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:42:33 2017

@author: Fan
"""


import pandas as pd
import numpy as np

def load_data2(train_data,test_data):


    all_user = np.union1d(train_data["User"],test_data["User"])
    all_movie = np.union1d(train_data["Movie"],test_data["Movie"])
    n_users = len(all_user)
    n_items = len(all_movie)

    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[np.where(all_user==line[2])[0]-1, np.where(all_movie == line[1])[0]-1] = line[3]
    
    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[np.where(all_user==line[2])[0]-1, np.where(all_movie == line[1])[0]-1] = line[3]
    
    train2_df = pd.DataFrame(train_data_matrix, index = all_user, columns = all_movie)
    test2_df = pd.DataFrame(test_data_matrix, index = all_user, columns = all_movie)
    
    return (train2_df,test2_df)
    








