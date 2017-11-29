#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:25:37 2017

@author: Chloe
"""

def z_score2(data_train,data_test,weights_neighbor,neighbors):
    import pandas as pd
    import numpy as np
    
    user1 = data_test['User'] # all the users in test data
    user2 = np.array(user1)   # transform to np
    user_id = np.array(data_train.index) # all the users in training data
    
    sorter = np.argsort(user_id)
    user_loc = np.searchsorted(user_id, user2, sorter = sorter) 
   
    movie1 = data_test['Movie'] # all the movies in test data
    movie2 = np.array(movie1)   
    movie_id = data_train.columns.values.astype(int) # all the users in training data
     
    sorter2 = np.argsort(movie_id)
    movie_loc = np.searchsorted(movie_id, movie2, sorter = sorter2)
    
    p = []
    for a,i in zip(user_loc, movie_loc):
        # a is the test user location, i is the test movie location 
        l = data_train.values[a,:] # l contains a user's rating information 
        m = data_train.values[neighbors[a,:],] # m contains a's neighbors rating information
        n = data_train.values[neighbors[a,:],i] # n is the neighbors' rating information for movie i 
        
        ra_mean = l[l.nonzero()].mean() 
        ra_std = l[l.nonzero()].std()
        r_ui = n[n.nonzero()] 
        ru = m[n.nonzero()] # all the ru information which rates the i movie
        ru[ru == 0] = np.nan
        ru_mean = np.nanmean(ru, axis = 1) # mean value for ru mean
        ru_std = np.nanstd(ru, axis = 1)
        
       
        if len(ru) == 0:
            p.append(ra_mean)
        else:
            p.append(ra_mean + ra_std * (np.sum((r_ui - ru_mean)/ru_std * weights_neighbor[a,n.nonzero()]))/np.sum(weights_neighbor[a,n.nonzero()]))
    
        # transform the output to a dataframe
        test_users = np.unique(user2)
        test_movies = np.unique(movie2)
        n_test_users = len(test_users)
        n_test_movies = len(test_movies)
        
        pred2 = np.zeros((n_test_users, n_test_movies))
        
        for line in data_test.itertuples():
            pred2[np.where(test_users==line[2])[0]-1, np.where(test_movies == line[1])[0]-1] = line[4]
        
        pred2 = pd.DataFrame(pred2,index = test_users, columns = test_movies)
        
    return pred2


#########################Deviation for mean

def Deviation_for_mean2(data_train, data_test, weights_neighbor, neighbors):
    
    import pandas as pd
    import numpy as np
    
    
    user1 = data_test['User'] # all the users in test data
    user2 = np.array(user1)   # transform to np
    user_id = np.array(data_train.index) # all the users in training data
    
    sorter = np.argsort(user_id)
    user_loc = np.searchsorted(user_id, user2, sorter = sorter) 
   
    movie1 = data_test['Movie'] # all the movies in test data

    movie2 = np.array(movie1)   
    movie_id = data_train.columns.values.astype(int) # all the users in training data
     
    sorter2 = np.argsort(movie_id)
    movie_loc = np.searchsorted(movie_id, movie2, sorter = sorter2)

    
    p = []
    for a,i in zip(user_loc, movie_loc):
        # a is the test user location, i is the test movie location 
        l = data_train.values[a,:] # l contains a user's rating information 
        m = data_train.values[neighbors[a,:],] # m contains a's neighbors rating information
        n = data_train.values[neighbors[a,:],i] # n is the neighbors' rating information for movie i 
        
        ra_mean = l[l.nonzero()].mean() 
        r_ui = n[n.nonzero()] 
        ru = m[n.nonzero()] # all the ru information which rates the i movie
        ru[ru == 0] = np.nan
        ru_mean = np.nanmean(ru, axis = 1) # mean value for ru mean
        
        if len(ru) == 0:
            p.append(ra_mean)
        else:
            p.append(ra_mean + (np.sum((r_ui - ru_mean) * weights_neighbor[a,n.nonzero()]))/np.sum(weights_neighbor[a,n.nonzero()]))
    
    data_test.loc[:,'Prediction'] = p
    
    
    test_users = np.unique(user2)
    test_movies = np.unique(movie2)
    n_test_users = len(test_users)
    n_test_movies = len(test_movies)
    
    pred2 = np.zeros((n_test_users, n_test_movies))
    
    for line in data_test.itertuples():
        pred2[np.where(test_users==line[2])[0]-1, np.where(test_movies == line[1])[0]-1] = line[4]
    
    pred2 = pd.DataFrame(pred2,index = test_users, columns = test_movies)
    
    return pred2
