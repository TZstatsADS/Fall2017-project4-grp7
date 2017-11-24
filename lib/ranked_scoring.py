#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:37:08 2017

@author: Fan
"""

# Function to calculate the prediction ranked scoring
def ranked_score(test_df, predict_df, d = 0.5, alpha = 5):
    '''
    This function aims to calculate the ranked scoring of our prediction by using
    implicit voting data
    
    Parameter
    ---------
    test_df : A pandas dataframe that contains all the information about the test 
    active users and their voting. 
    predict_df : A pandas dataframe that contians all the prediction information
    d : d is the neutral vote, which used to effectively remove those unavailable items
    alpha : Alpha is the viewing halflife, default value is 5.
    
    Return
    ------
    R : A number represents the ranked scoring 
    '''
    
    import numpy as np
    import pandas as pd
    
    
    m,n = test_df.shape # m is the number of test users, n is the number of items
    Ra = 0 # initial value for Ra
    Ra_max = 0 # initial value for Ra_max
    
    
    # For each of the user, we need to calculate the Ra and Ra_max and then sum them up
    for i in np.arange(m):
        user = test_df.index[i] # get the user name
        
        test_ol = test_df.loc[user,:] # test list
        predict_ol = predict_df.loc[user,:] # predict list
        predict_test_combine = pd.concat([predict_ol ,test_ol], axis = 1)  # join above two list
        predict_test_combine.columns = ['pred_user', 'test_user'] # change columns names
        predict_test_combine = predict_test_combine.dropna(how='any') # delete NA
        predict_test_combine = predict_test_combine.sort_values(by = ['pred_user'], ascending = False) # Order by prediction list
        
        for j in np.arange(n):
        
            # Calculate Ra for each of the item for this user
            Ra_j = max(predict_test_combine.iloc[j,1] - d, 0)/2 ** (j/(alpha - 1)) # j start from 0
            Ra = Ra + Ra_j
            
            # Calculate Ra_max for each of the item for this user
            Ra_max_i = max(test_ol[j] - d, 0)/2 ** (j/(alpha - 1))
            Ra_max = Ra_max + Ra_max_i
        
    # calculate the ranked score based on its defination
    R = 100 * Ra/Ra_max
    
    return R

    
    
    
    
    
    
    
    
    
    


