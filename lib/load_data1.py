import pandas as pd
import numpy as np




def load_data1(ms_data):
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

