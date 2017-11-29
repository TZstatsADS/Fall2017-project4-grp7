def Deviation_for_mean1(train,test,weights_neighbor,neighbors):
    import numpy as np
    import pandas as pd
    
    row_index=train.index
    col_index=train.columns
    train=train.values #transfer to numpy
    m = len(train) # number of all users
    num_neighbors=weights_neighbor.shape[1] # number of neighbors we select
    prediction=[]
    for user,neighbor,weight_neighbor in zip(train,neighbors,weights_neighbor):
        neighbors_rating=train[neighbor]
        neighbors_mean=np.mean(neighbors_rating,axis=1)
        neighbors_rating=neighbors_rating-np.mean(neighbors_rating,axis=1).reshape(num_neighbors,1)
        
        nominator=np.sum(neighbors_rating*weight_neighbor.reshape(num_neighbors,1),axis=0)
        denominator=np.sum(weight_neighbor)
        
        user_result=np.array(nominator)/denominator+np.mean(user)
        prediction.append(user_result)
    prediction=np.array(prediction)
    
    # transfer to pandas:
    prediction=pd.DataFrame(prediction)
    prediction.index=row_index
    prediction.columns=col_index
    
    return prediction.loc[test.index,test.columns]


def z_score1(train,test,weights_neighbor,neighbors):
    import numpy as np
    import pandas as pd
    
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



