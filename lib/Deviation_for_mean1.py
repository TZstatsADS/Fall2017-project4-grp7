
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

if __name__ == '__main__':
    test1 = pd.read_csv("../output/test1_df.csv",index_col = 0)
    train1 = pd.read_csv("../output/train1_df.csv",index_col = 0)
    with open("../output/PearsonCosineMatrix/pearson_correlation1.pkl", 'rb') as f1:
        pearson_corr1 = pickle.load(f1)
    Deviation_for_mean1(train1 = train1,test1 = test1, n = 10， similarity_weights = pearson_corr1)
