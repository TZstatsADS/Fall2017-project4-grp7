def z_score1(train,test,weights_neighbor,neighbors):
    row_index=train.index
    col_index=train.columns
    train=train.values #transfer to numpy
    m=len(train) # number of all users
    num_neighbors=weights_neighbor.shape[1] # number of neighbors we select
    prediction=[]
    for user,neighbor,weight_neighbor in zip(train,neighbors,weights_neighbor):
        user_sigma=np.std(user)
        neighbors_rating=train[neighbor]
        neighbors_sigma= np.std(neighbors_rating,axis=1)
        neighbors_mean=np.mean(neighbors_rating,axis=1)
        neighbors_rating=neighbors_rating-np.mean(neighbors_rating,axis=1).reshape(num_neighbors,1)
        
        nominator=np.sum(neighbors_rating*weight_neighbor.reshape(num_neighbors,1)/neighbors_sigma.reshape(num_neighbors,1),axis=0)
        denominator=np.sum(weight_neighbor)
        
        user_result=user_sigma*np.array(nominator)/denominator+np.mean(user)
        prediction.append(user_result)
    prediction=np.array(prediction)
    
    # transfer to pandas:
    prediction=pd.DataFrame(prediction)
    prediction.index=row_index
    prediction.columns=col_index
    
    return prediction.loc[test.index,test.columns]
"""
def z_score1(train1,test1,similarity_weights,n = 10):
    import numpy as np
    import pandas as pd
    weights_neighbor1 = dict((key,(np.array(similarity_weights.loc[key,:].sort_values(ascending = False).index[1:n+1]),np.array(pearson_corr1.loc[key,:].sort_values(ascending = False)[1:n+1]))) for key in similarity_weights.index)
    r_a_mean = np.array(train1.loc[test1.index].mean(axis=1))
    sigma_a =  np.array(train1.loc[test1.index].std(axis=1))
    index_a_u, w_a_u = zip(*[(weights_neighbor1[k][0], weights_neighbor1[k][1]) for k in test1.index])
    r_u_mean = np.array([train1.loc[index_a_u[i]].mean(axis=1) for i in range(len(index_a_u))])
    sigma_u = np.array([train1.loc[index_a_u[i]].std(axis=1) for i in range(len(index_a_u))])
    p_a_i = pd.DataFrame(0, index = test1.index, columns = test1.columns)
    for a in range(len(test1.index)):
        for i in test1.iloc[a][test1.iloc[a]==1].index:
            r_u_i = np.array(train1.loc[index_a_u[a],i])
            p_a_i.loc[test1.index[a],i] = r_a_mean[a]+sigma_a[a]*((r_u_i- r_u_mean[a])/sigma_u[a]).dot(w_a_u[a])/(w_a_u[a].sum())
    return  p_a_i
"""

def z_score2(data_train,data_test,weights_neighbor,neighbors):
    user = data_test['User']
    user = np.array(user)
    user_id = np.array(data_train.index)
    user_id = user_id.searchsorted(user)
    movie = data_test['Movie']
    movie = np.array(movie)
    movie_id = data_train.columns.values.astype(int)
    movie_id = movie_id.searchsorted(movie)

    p = []
    for a,i in zip(user_id, movie_id):
        p.append(np.mean(data_train.values[a,:]) + np.std(data_train.values[a,:])*np.sum((data_train.values[neighbors[a,:],i-1] - np.mean(data_train.values[neighbors[a,:],]))/np.std(data_train.values[neighbors[a,:],])*weights_neighbor[a,:])/np.sum(weights_neighbor[a,:]))
    return(p)
