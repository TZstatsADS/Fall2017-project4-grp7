
def Deviation_for_mean1(train1,test1, n = 10，similarity_weights):

    import numpy as np
    import pandas as pd
    import pickle

    weights_neighbor1 = dict((key,(np.array(pearson_corr1.loc[key,:].sort_values(ascending = False).index[1:n+1]),np.array(pearson_corr1.loc[key,:].sort_values(ascending = False)[1:n+1]))) for key in similarity_weights.index)

    r_a_mean = np.array(train1.loc[test1.index].mean(axis=1))
    index_a_u, w_a_u = zip(*[(weights_neighbor1[k][0], weights_neighbor1[k][1]) for k in test1.index])
    r_u_mean = np.array([train1.loc[index_a_u[i]].mean(axis=1) for i in range(len(index_a_u))])
    p_a_i = pd.DataFrame(0, index = test1.index, columns = test1.columns)
    for a in range(len(test1.index)):
        for i in test1.iloc[a][test1.iloc[a]==1].index:
            r_u_i = np.array(train1.loc[index_a_u[a],i])
            p_a_i.loc[test1.index[a],i] = r_a_mean[a] + (r_u_i- r_u_mean[a]).dot(w_a_u[a])/w_a_u[a].sum()
    p_a_i.to_csv("./output/Deviation_for_mean1.csv")
    return  p_a_i

def Deviation_for_mean2(data_train,data_test,weights_neighbor,neighbors):
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
        p.append(np.mean(data_train.values[a,:]) + np.sum((data_train.values[neighbors[a,:],i] - np.mean(data_train.values[neighbors[a,:],]))*weights_neighbor[a,:])/np.sum(weights_neighbor[a,:]))
    return p

    
def main():
    test1 = pd.read_csv("../output/test1_df.csv",index_col = 0)
    train1 = pd.read_csv("../output/train1_df.csv",index_col = 0)
    with open("../output/PearsonCosineMatrix/pearson_correlation1.pkl", 'rb') as f1:
        pearson_corr1 = pickle.load(f1)
    pred_dfm1=Deviation_for_mean1(train1 = train1,test1 = test1, n = 10， similarity_weights = pearson_corr1)
    
    
    
    train2 = pd.read_csv('./output/train2_df.csv',index_col = 0)
    test_2 = pd.read_csv('./eachmovie_sample/data_test.csv',index_col = 0)
    pred_dfm2 = Deviation_for_mean2(train2,test_2,weights_neighbor2,neighbors2)

if __name__ == '__main__':
    main()
    
    
    
    