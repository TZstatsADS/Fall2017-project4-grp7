#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:25:37 2017

@author: Chloe
"""

#########################Deviation for mean
def Deviation_for_mean2(data_train,data_test,weights_neighbor,neighbors):
    import pandas as pd
    import numpy as np
    
    user1 = data_test['User']
    user2 = np.array(user1)   
    user_id = np.array(data_train.index)
    user_id = user_id.searchsorted(user2)
    movie1 = data_test['Movie']
    movie2 = np.array(movie1)   
    movie_id = data_train.columns.values.astype(int)
    movie_id = movie_id.searchsorted(movie2)
    
    p = []
    for a,i in zip(user_id, movie_id):
        # a is the test user location, i is the test movie location 
        l = data_train.values[a,:] # l contains a user's rating information 
        m = data_train.values[neighbors[a,:],] # m contains a's neighbors rating information
        n = data_train.values[neighbors[a,:],i] # n is the neighbors' rating information for movie i 
        
        ra_mean = l[l.nonzero()].mean() 
        r_ui = n[n.nonzero()] 
        ru = m[n.nonzero()] # all the ru information which rates the i movie
        ru[ru == 0] = np.nan
        ru_mean = np.nanmean(ru, axis = 1) # mean value for ru mean
        
       
        p.append(ra_mean + (np.sum((r_ui - ru_mean) * weights_neighbor[a,n.nonzero()]))/np.sum(weights_neighbor[a,n.nonzero()]))
    
    return p
