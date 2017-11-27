#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:46:45 2017

@author: Fan
"""


# function to calculate the cosine similarities for implicit data
def cosineSimi1(train1_df):
    '''
    This function aims to calcualte the cosine similarity for implicit voting data
    
    Parameter
    ---------
    train1_df: A pandas dataframe contains implicit voting data
    
    Rutern
    ------
    cosine_matrix: A pandas dataframe that the dimension is n * n where n 
    is the number of users.
    '''
    
    import pandas as pd
    import numpy as np
    from numpy import linalg as LA
    from datetime import datetime
    import itertools
    
    # Get information from the dataset
    user_name = train1_df.index
    rows = train1_df.shape[0]
    
    # Transform to np.array to make it faster
    train1_df = np.array(train1_df)
    
   # Calculate time
    time_list = []
    time_list.append(datetime.now())
    
    # Initial correlation matrix
    correlations = np.identity(rows)
    
    for a, i in itertools.combinations(np.arange(rows),2):
        # a,i are two different users's row index
        
        if rows % 300 == 0:
          time_list.append(datetime.now())
          print("a = "+str(a)+"\tb = "+str(i)+"\n\t"+str((time_list[-1] - time_list[1])))
      
        va = train1_df[a] # The first user's voting 
        vi= train1_df[i] # The second user's voting
        
        part1 = va.dot(vi) # The numerator part
        if part1 != 0:
            part2 = LA.norm(va) * LA.norm(vi)
            
            correlations[a,i] = part1/part2
            correlations[i,a] = correlations[a,i]
    
    cosine_matrix = pd.DataFrame(correlations, index = user_name, columns = user_name)

    return cosine_matrix




# function to calculate the cosine similarities for explicit data
def cosineSimi2(train2_df):
    '''
    This function aims to calcualte the cosine similarity for explicit voting data
    
    Parameter
    ---------
    train2_df: A pandas dataframe contains explicit voting data
    
    Rutern
    ------
    cosine_matrix: A pandas dataframe that the dimension is n * n where n 
    is the number of users.
    '''
    
    import pandas as pd
    import numpy as np
    from numpy import linalg as LA
    from datetime import datetime
    import itertools
    
    # Get information from the dataset
    user_name = train2_df.index
    rows = train2_df.shape[0]
    
    # Transform to np.array to make it faster
    train2_df = np.array(train2_df)
    
   # Calculate time
    time_list = []
    time_list.append(datetime.now())
    
    # Initial correlation matrix
    correlations = np.identity(rows)
    
    for a, i in itertools.combinations(np.arange(rows),2):
        # a,i are two different users's row index
        
        if rows % 300 == 0:
          time_list.append(datetime.now())
          print("a = "+str(a)+"\tb = "+str(i)+"\n\t"+str((time_list[-1] - time_list[1])))
     
        
        interIndex = np.nonzero(train2_df[a] * train2_df[i])
        if interIndex[0].size != 0: # if there is a item that both user has rated
            va = train2_df[a][interIndex] # The first user's voting 
            vi= train2_df[i][interIndex] # The second user's voting
        
            part1 = va.dot(vi) # The numerator part
            if part1 != 0:
                part2 = LA.norm(va) * LA.norm(vi)
                
                correlations[a,i] = part1/part2
                correlations[i,a] = correlations[a,i]
    
    cosine_matrix = pd.DataFrame(correlations, index = user_name, columns = user_name)

    return cosine_matrix







