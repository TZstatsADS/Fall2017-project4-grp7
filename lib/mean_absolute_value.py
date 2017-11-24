#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:04:51 2017

@author: Fan
"""


def mean_absolute_error(test_df, predict_df):
    '''
    This function aims to calculate the mean absolute error of our prediction
    
    Parameter
    ---------
    test_df : A pandas dataframe that contains all the information about the test 
    active users and their voting. 
    predict_df : A pandas dataframe that contians all the prediction information
    
    Return
    ------
    MAE : A number represents the mean absolute error
    '''
    
    import pandas as pd
    import numpy as np
    
    # Preprocess the data
    test_df = test_df.replace(0, np.nan) # replace 0 with na
    test_df = test_df.dropna(axis = [0,1], how = 'all') # delete the rows and columns that are all NAs
    
    m,_ = test_df.shape # m is the number of the test user
    MAE = 0 # initial mean absolute value
    
    # For each of the user, we need to calculate the absolute difference 
    for i in np.arange(m):
        
        user = test_df.index[i] # user name
        item = test_df.loc[user, :].dropna() # item list 
        item_name = item.index # voted item name
        
        predict_item = predict_df.loc[user, list(item_name)] # predict voted value
        
        MAE = MAE + np.abs(predict_item - item).sum()/len(item) # calculate the sum of MAE
        
    
    MAE = MAE/m # Calculate the mean of absolute value
        
        
    return MAE
         


