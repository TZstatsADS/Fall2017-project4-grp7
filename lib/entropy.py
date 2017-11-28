#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:12:49 2017

@author: Fan
"""

# function to calculate the difference entropy similarity for data1
def entropySimi1(train1_df):
    '''
    This function aims to calcualte the difference entropy similarity for implicit voting data
    
    Parameter
    ---------
    train1_df: A pandas dataframe contains implicit voting data
    
    Rutern
    ------
    entropy_matrix: A pandas dataframe that the dimension is n * n where n 
    is the number of users. 
    '''
    
    import pandas as pd
    import numpy as np
    from scipy import stats
    from datetime import datetime
    import itertools
    
    # Get information from the dataset
    user_name = train1_df.index
    rows = train1_df.shape[0]
    
    # Transform to np.array to make it faster
    train1_df = np.array(train1_df, dtype = float)
    
   # Calculate time
    time_list = []
    time_list.append(datetime.now())
    
    # Initial correlation matrix
    correlations = np.identity(rows)
    
    for a, i in itertools.combinations(np.arange(rows),2):
        # a,i are two different users's row index
        
        if a % 300 == 0 and i % 300 == 0:
          time_list.append(datetime.now())
          print("a = "+str(a)+"\tb = "+str(i)+"\n\t"+str((time_list[-1] - time_list[1])))
      
        # calculate the difference score
        diff = np.abs(train1_df[a] - train1_df[i])
        # calculate the pdf of the difference score
        diff_prob = stats.itemfreq(diff)
        diff_prob[:,1]  = diff_prob[:,1]/len(diff) # first column is the element and second column is the prob
        Pd = diff_prob[:,1]
        Pd_score = diff_prob[:,0]
        # calculate weighted entropy 
        H = -Pd.dot(np.log(Pd) * (Pd_score))
        #normalize to [0,1]; The smaller this value, the similier the users.
        H_scaled = 1/(1 + np.exp(-H))
        # Final score; The larger the value, the similier the users.
        score = 1 - H_scaled
        
        correlations[a,i] = score
        correlations[i,a] = correlations[a,i]
    
    entropy_matrix = pd.DataFrame(correlations, index = user_name, columns = user_name)

    return entropy_matrix


# function to calculate the difference entropy similarity for data2
def entropySimi2(train2_df):
    '''
    This function aims to calcualte the difference entropy similarity for explicit voting data
    
    Parameter
    ---------
    train2_df: A pandas dataframe contains explicit voting data
    
    Rutern
    ------
    entropy_matrix: A pandas dataframe that the dimension is n * n where n 
    is the number of users. 
    '''
    
    import pandas as pd
    import numpy as np
    from scipy import stats
    from datetime import datetime
    import itertools
    
    # Get information from the dataset
    user_name = train2_df.index
    rows = train2_df.shape[0]
    
    # Transform to np.array to make it faster
    train2_df = np.array(train2_df, dtype = float)
    
   # Calculate time
    time_list = []
    time_list.append(datetime.now())
    
    # Initial correlation matrix
    correlations = np.identity(rows)
    
    for a, i in itertools.combinations(np.arange(rows),2):
        # a,i are two different users's row index
        
        if a % 300 == 0 and i % 300 == 0:
          time_list.append(datetime.now())
          print("a = "+str(a)+"\tb = "+str(i)+"\n\t"+str((time_list[-1] - time_list[1])))
      
        
        interIndex = np.nonzero(train2_df[a] * train2_df[i])
        if interIndex[0].size != 0: # if there is a item that both user has rated
        
            # calculate the difference score
            diff = np.abs(train2_df[a][interIndex] - train2_df[i][interIndex])
            # calculate the pdf of the difference score
            diff_prob = stats.itemfreq(diff)
            diff_prob[:,1]  = diff_prob[:,1]/diff.shape[0] # first column is the element and second column is the prob
            Pd = diff_prob[:,1]
            Pd_score = diff_prob[:,0]
            # calculate weighted entropy 
            H = -Pd.dot(np.log(Pd) * (Pd_score))
            #normalize to [0,1]; The smaller this value, the similier the users.
            H_scaled = 1/(1 + np.exp(-H))
            # Final score; The larger the value, the similier the users.
            score = 1 - H_scaled
            
            correlations[a,i] = score
            correlations[i,a] = correlations[a,i]
    
    entropy_matrix = pd.DataFrame(correlations, index = user_name, columns = user_name)

    return entropy_matrix





