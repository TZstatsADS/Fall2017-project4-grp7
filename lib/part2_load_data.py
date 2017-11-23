import pandas as pd
import numpy as np

ms_train = pd.read_csv('../data/MS_sample/data_train.csv',usecols=range(1,4))
ms_test = pd.read_csv('../data/MS_sample/data_test.csv',usecols=range(1,4))


def sparse2dense1(ms_data):
    df_ms_data = pd.DataFrame(0,index=ms_data[ms_data["V1"]=="C"]["V2"].unique(),columns=ms_data[ms_data["V1"]=="V"]["V2"].unique())
    i=0
    while i < ms_data.shape[0]:
        if ms_data["V1"][i]=="C":
            row_index = ms_data["V2"][i]
            i+=1
            while ms_data["V1"][i]=="V":
                col_index = ms_data["V2"][i]
                df_ms_data.loc[row_index,col_index] += 1
                i += 1
                if i == ms_data.shape[0]:
                    break         
 
    return df_ms_data

ms_train_pd = sparse2dense1(ms_train)
ms_test_pd =  sparse2dense1(ms_test)

ms_train_np=ms_train_pd.as_matrix()
ms_test_np=ms_test_pd.as_matrix()

#print ms_train_pd.columns
#print ms_train_np.shape

#np.save('../output/train_matrix',ms_train_np)
#np.save('../output/test_matrix',ms_test_np)

#ms_train_pd.to_csv('../output/train_matrix.csv',header=True,index=True)
#ms_test_pd.to_csv('../output/test_matrix.csv',header=True,index=True)