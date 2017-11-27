# -*- coding: utf-8 -*-
import requests
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.stats.stats import pearsonr
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
from numpy import matrix

from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4) 

##############SimRank data1################
ms_train = pd.read_csv('./Desktop/ADS/fall2017-project4-group7/data/MS_sample/data_train.csv', sep='\s+').astype(str)

n_row=len(np.unique(ms_train[ms_train[:,0]=="C",1])) 
m_col=len(np.unique(ms_train[ms_train[:,0]=="V",1]))
df = pd.DataFrame(0,index=np.unique(ms_train[ms_train[:,0]=="C",1]),columns=np.unique(ms_train[ms_train[:,0]=="V",1]))
# error
i=0
while i < data.shape[0]:
    if data[i,0]=="C":
        row_index = data[i,1]
        i+=1
        while data[i,0]=="V":
            col_index = data[i,1]
            df.loc[row_index,col_index] += 1
            i += 1
            if i == data.shape[0]:
                break
            
#df.to_csv('./data/matrix.csv')
#df_votes = pd.read_csv('./data/matrix.csv')
df_vote = df.values


case = list(np.unique(ms_train[ms_train[:,0]=="C",1]))
vote = list(np.unique(ms_train[ms_train[:,0]=="V",1]))

# Graph the relation num
graph = numpy.matrix(numpy.zeros([n_row, m_col]))

for item in df_vote:
    case = item[0]
    vote = item[1]
    c_i = case.index(case)
    v_j = vote.index(vote)
    graph[c_i, v_j] += 1

print graph

case_sim = matrix(numpy.identity(n_row))
vote_sim = matrix(numpy.identity(m_col))

def get_vote_num(case):
    c_i = case.index(case)
    return graph[c_i]

def get_case_num(vote):
    v_j = vote.index(vote)
    return graph.transpose()[v_j]

def get_vote(case):
    series = get_vote_num(case).tolist()[0]
    return [ vote[x] for x in range(len(series)) if series[x] > 0 ]

def get_case(vote):
    series = get_case_num(vote).tolist()[0]
    return [ case[x] for x in range(len(series)) if series[x] > 0 ]


def case_simrank(c1, c2, C):
    """
    graph[c_i] connect to vote
    print "c1.vote"
    print get_vote_num(c1).tolist()
    print "c2.vote"
    print get_vote_num(c2).tolist()
    """
    if c1 == c2 : return 1
    prefix = C / (get_vote_num(c1).sum() * get_vote_num(c2).sum())
    postfix = 0
    for v_i in get_vote(c1):
        for v_j in get_vote(c2):
            i = vote.index(v_i)
            j = vote.index(v_j)
            postfix += vote_sim[i, j]
    return prefix * postfix
    

def vote_simrank(v1, v2, C):
    """
    graph should be transposed to make vote to be index
    print "v1.case"
    print get_case_num(v1)
    print "v2.case"
    print get_case_num(v2)
    """
    if v1 == v2 : return 1
    prefix = C / (get_case_num(v1).sum() * get_case_num(v2).sum())
    postfix = 0
    for c_i in get_case(v1):
        for c_j in get_case(v2):
            i = case.index(c_i)
            j = case.index(c_j)
            postfix += case_sim[i,j]
    return prefix * postfix


def simrank(C=0.8, times=1):
    global case_sim, vote_sim

    for run in range(times):
        # case_simrank
        new_case_sim = matrix(numpy.identity(n_row))
        for ci in case:
            for cj in case:
                i = case.index(qi)
                j = case.index(qj)
                new_case_sim[i,j] = case_simrank(ci, cj, C)

        # vote_simrank
        new_vote_sim = matrix(numpy.identity(m_col))
        for vi in vote:
            for aj in vote:
                i = vote.index(vi)
                j = vote.index(vj)
                new_vote_sim[i,j] = vote_simrank(vi, vj, C)

        case_sim = new_case_sim
        vote_sim = new_vote_sim


if __name__ == '__main__':
    print case
    print vote
    simrank()
    print case_sim
    print vote_sim