#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:42:33 2017

@author: Fan
"""

import os
import pandas as pd
import numpy as np

os.chdir('/Users/gongfan/Documents/GitHub/fall2017-project4-group7/lib')
os.getcwd()

train_data = pd.read_csv('../data/eachmovie_sample/data_train.csv',usecols=["Movie","User","Score"])
test_data  = pd.read_csv('../data/eachmovie_sample/data_test.csv',usecols=["Movie","User","Score"])

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

train2_df.to_csv('../output/train2_df.csv',header=True,index=True)
test2_df.to_csv('../output/test2_df.csv', header=True, index=True)







