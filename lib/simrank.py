def simrank(dense_mat, maxIteration = 30, C1 = 0.8):
    '''
    Parameter
    ---------
    dense_mat: has the shape of # of users * # of movies.
    maxIteration : is the # of iteration for simrank to converage usually the algrithm converges fast.
    C1: is the rate of decay
    '''

    import numpy as np
    import pandas as pd
    n_users, n_movies = dense_mat.shape
    nodesnum = n_users + n_movies
    trans_mat = np.zeros((nodesnum,nodesnum))
    trans_mat[0:n_users,n_users:nodesnum]= dense_mat
    trans_mat[n_users:nodesnum,0:n_users] = dense_mat.T
    trans_mat = trans_mat/trans_mat.sum(axis=0)[None,:]
    sim_mat = np.identity(nodesnum) * (1 - C1)
    for i in range(maxIteration):
        sim_mat = C1 * np.dot(np.dot(trans_mat.transpose(),sim_mat), trans_mat) + (1 - C1) * np.identity(nodesnum)
        sim_mat[sim_mat < 0.001] = 0
        if i == (maxIteration-1):
           sim_mat = sim_mat[0:n_users,0:n_users]
    return sim_mat

if __name__ == '__main__':
    train_data = pd.read_csv('./data/eachmovie_sample/data_train.csv',usecols=["Movie","User","Score"])
    test_data  = pd.read_csv('./data/eachmovie_sample/data_test.csv',usecols=["Movie","User","Score"])

    all_user = np.union1d(train_data["User"],test_data["User"])
    all_movie = np.union1d(train_data["Movie"],test_data["Movie"])
    n_users = len(all_user)
    n_items = len(all_movie)

    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[np.where(all_user==line[2])[0]-1, np.where(all_movie == line[1])[0]-1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[np.where(all_user==line[2])[0]-1, np.where(all_movie == line[1])[0]-1] = line[3]

    simRank_mat = simrank(dense_mat = train_data_matrix, maxIteration = 30, C1 = 0.8)
