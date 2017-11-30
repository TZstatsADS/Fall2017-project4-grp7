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



def neighbor_size_dfm1(data_train,data_test,similarity_weights):
    import matplotlib.pyplot as plt
    i_range = np.arange(start = 1, stop = 10, step = 1)
    score_list_ = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values)
        pred_dfm  = Deviation_for_mean1(data_train,data_test,weights_neighbor,neighbors)
        pred_dfm = pd.DataFrame(pred_dfm)
        eva= ranked_score(data_test, pred_dfm, d = 0.1, alpha = 5)
        score_list.append(eva)
        print(i)
        n_opt = score_list.index(max(score_list))+1
    return score_list, n_opt


def neighbor_size_zscore1(data_train,data_test,similarity_weights):
    import matplotlib.pyplot as plt
    i_range = np.arange(start = 1, stop = 10, step = 1)
    score_list = []
    for i in i_range:
        weights_neighbor, neighbors = selecting_neighborhood(i, similarity_weights, data_train.values)
        pred_zscore  = z_score1(data_train,data_test,weights_neighbor,neighbors)
        pred_zscore  = pd.DataFrame(pred_zscore)
        eva = ranked_score(data_test, pred_zscore, d = 0.1, alpha = 5)
        score_list.append(eva)
        print(i)
        n_opt = score_list.index(max(score_list))+1
    return score_list, n_opt

##########################ranked score################################
score_list_list = []
n_opt_list = []
for similairty in pearson_correlation1, cosine_correlation1, entropy_correlation1:
    score_list, n_opt = neighbor_size_dfm1(train1,test1,similairty)
    print("similarity done")
    score_list_list.append(score_list)
    n_opt_list.append(n_opt)


score_list_list_z = []
n_opt_list_z = []
for similairty in pearson_correlation1, cosine_correlation1, entropy_correlation1:
    score_list, n_opt = neighbor_size_zscore1(train1,test1,similairty)
    print("similarity done")
    score_list_list_z.append(score_list)
    n_opt_list_z.append(n_opt)


######################data1, Deviation for Mean, n=13#########################
pearson_corr1_rank_score1 = score_list_list[0]
cosine_corr1_rank_score1 = score_list_list[1]
entropy_corr1_rank_score1 = score_list_list[2]

######################data1, z_score, n=13#########################
pearson_corr1_rank_score1 = score_list_list_z[0]
cosine_corr1_rank_score1 = score_list_list_z[1]
entropy_corr1_rank_score1 = score_list_list_z[2]
