#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 22:04:40 2017

@author: Fan
"""


def roc_sensitivity(test_df, predict_df, goodness = 4):
    '''
    This function aims to calculate the ROC sensitivity of our prediction
    
    Parameter
    ---------
    test_df : A pandas dataframe that contains all the information about the test 
    active users and their voting. 
    predict_df : A pandas dataframe that contians all the prediction information.
    goodness : A number determines which items are 'good' or 'bad'. By default we choose 4, 
    which means the items whose rating larger than and equal to 4 will be considered as 'good' 
    and otherwise will be considered as 'bad'
    
    
    Return
    ------
    sensitivity : A number represents the ROC sensitivity
    '''
    
    import pandas as pd
    import numpy as np
    
    # Preprocess the data 
    test_df = test_df.replace(0, np.nan) # replace 0 with na
    test_df = test_df.dropna(axis = [0,1], how = 'all') # delete the rows and columns that are all NAs

    
    m,_ = test_df.shape # m is the number of the test user, and n is the number of items
    num_TP = 0 # number of true positive value
    num_actual_good = 0 # number of actual good value
    
    
    for i in np.arange(m):
        
        user = test_df.index[i] # user name
        item = test_df.loc[user, :].dropna() # item list 
        item_name = item.index.astype(int) # voted item name
        
        predict_item = predict_df.loc[user, list(item_name)] # predict voted value
        
        # Transform the test data and the prediction data into 'good' and 'bad'
        item_bool = item >= goodness
        item[item_bool] = 'good'
        item[~item_bool] = 'bad'
        
        predict_item_bool = predict_item >= goodness
        predict_item[predict_item_bool] = 'good'
        predict_item[~predict_item_bool] = 'bad'
        
        for j in np.arange(len(item)):
            item_j = item[j]
            predict_item_j = predict_item.values[j]
            
            if item_j == 'good' and predict_item_j == 'good':
                num_actual_good += 1
                num_TP += 1
            elif item_j == 'good' and predict_item_j == 'bad':
                num_actual_good += 1
            else:
                continue
        
    sensitivity = num_TP/num_actual_good
    return sensitivity
                
            
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    