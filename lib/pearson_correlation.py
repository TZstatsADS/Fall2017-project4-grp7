#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:18:16 2017

@author: Fan
"""


# function to calculate the Pearson Correlation for data1

def pearsonSimi1(train1_df):
    '''
    This function aims to calcualte the correlation for implicit voting data
    
    Parameter
    ---------
    train1_df: A pandas dataframe contains implicit voting data
    
    Rutern
    ------
    correlation_matrix: A pandas dataframe that the dimension is n * n where n 
    is the number of users. 
    '''

    import numpy as np
    import pandas as pd
    import itertools
    from datetime import datetime
    
    # Get information from the dataset
    user_name = train1_df.index
    rows = train1_df.shape[0]
    # Transform to np.array to make it faster
    train1_df = np.array(train1_df)
    # Initial correlation matrix
    correlations = np.identity(rows)
    
    # Calculate time
    time_list = []
    time_list.append(datetime.now())
    
    for a,i in itertools.combinations(np.arange(rows),2):
        # a,i are two different users's row index
        
        if rows % 300 == 0:  
            time_list.append(datetime.now())
            print("a = "+str(a)+"\tb = "+str(i)+"\n\t"+str((time_list[-1] - time_list[1])))
        
        va = train1_df[a] # The first user's voting 
        vi= train1_df[i] # The second user's voting
        va_mean = va.mean() # The mean vote for a user
        vi_mean = vi.mean() # The mean vote for i user
        
        # The numerator part of correlation
        part1 = (va - va_mean).dot(vi - vi_mean) 
        
        if part1 != 0:
            # The denominator part of correlation
            part2 = np.sqrt((va-va_mean).dot(va-va_mean) * (vi-vi_mean).dot(vi-vi_mean))
            correlations[a,i] = part1/part2
            correlations[i,a] = correlations[a,i]
    
    correlation_matrix = pd.DataFrame(correlations, index = user_name, columns = user_name)
            
    return correlation_matrix

  





# function to calculate the Pearson Correlation for data2

def pearsonSimi2(train2_df):
    '''
    This function aims to calcualte the correlation for explicit voting data
    
    Parameter
    ---------
    train2_df: A pandas dataframe contains explicit voting data
    
    Rutern
    ------
    correlation_matrix: A pandas dataframe that the dimension is n * n where n 
    is the number of users. 
    '''
    import numpy as np
    import pandas as pd
    import itertools
    from datetime import datetime
    
    # Get information from the dataset
    user_name = train2_df.index
    rows = train2_df.shape[0]
    # Transform to np.array to make it faster 
    train2_df = np.array(train2_df)
    
    # Initial correlation matrix
    correlations = np.identity(rows)
    
    # Calculate time
    time_list = []
    time_list.append(datetime.now())
    
    for a,i in itertools.combinations(np.arange(rows),2):
        # a,i are two different users's row index
        
        if rows % 300 == 0:  
            time_list.append(datetime.now())
            print("a = "+str(a)+"\tb = "+str(i)+"\n\t"+str((time_list[-1] - time_list[1])))
        
        interIndex = np.nonzero(train2_df[a] * train2_df[i])
        if interIndex[0].size != 0: # if there is a item that both user has rated
            
            va = train2_df[a][interIndex] # The first user's voting 
            vi= train2_df[i][interIndex] # The second user's voting
            va_mean = va.mean() # The mean vote for a user
            vi_mean = vi.mean() # The mean vote for i user
        
            # The numerator part of correlation
            part1 = (va - va_mean).dot(vi - vi_mean) 
        
            if part1 != 0:
                # The denominator part of correlation
                part2 = np.sqrt((va-va_mean).dot(va-va_mean) * (vi-vi_mean).dot(vi-vi_mean))
                correlations[a,i] = part1/part2
                correlations[i,a] = correlations[a,i]
    
    correlation_matrix = pd.DataFrame(correlations, index = user_name, columns = user_name)
            
    return correlation_matrix






