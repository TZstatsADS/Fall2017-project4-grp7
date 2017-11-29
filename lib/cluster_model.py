import numpy as np









def train_cluster_model(X,k):
    m,n=X.shape
    
    h=np.unique(X)
    """
    use EM algorithm to train multinomial mixture model
    X: user matrix with Xi for each row Xi=[vi1,vi2,...,vin]
    m: users
    n: items
    k: clusters
    h: labels for votes. It should be a numeric vector starting from 0-X
    PI: assignment matrix
    c: weights for cluster
    
    THETA : three dimensional array for all parameters (kxnxh) 
    THETA[k]: k th cluster parameters. It's a nxh matrix. Item in n th row and d th column means the probability of n th vote being d
    In THETA[k], sum of each row should be 1 
    
    """
    def likelihood(X,k,THETA):
        """
        This is used to calculate likelihood function P(Xi|thetaK)
        Xi means ith user. It is a vector: (Vi1,Vi2,...Vin). It records ith user's votes for n items. e.g (1,2,5,6,0,2,...1)
        k means kth cluster. The parameters for kth cluster is THETA[k]
        THETA: three dimensional array for all parameters as mentioned above.
        """
        # return probability multiplication of all users' votes in one cluster and return the vector
        # vector=(user1P,user2P,...)
        
        
        theta_k=THETA[k]
        theta_selected=theta_k[np.arange(theta_k.shape[0]),X]
        return theta_selected.prod(axis=1)
    # set initial values for cluster weights
    c=np.repeat(1.0/k,k)
    
    # randomly assign parameters for each cluster
    THETA=[]
    for i in range(k):
        randoms=np.random.random(n*len(h))
        theta_matrix=randoms.reshape(n,len(h)) # theta_matrix: THETA[K] nxh matrix
        theta_matrix=theta_matrix/theta_matrix.sum(axis=1).reshape(n,1) # for each item the sum of h should be 1
        THETA.append(theta_matrix)
    
    # create assign matrix A:
    A=np.zeros((m,k))
    
    log_likelihood=0
    log_likelihood_old=100
    while np.absolute(log_likelihood-log_likelihood_old)>0.0001:
        
        # E-step:
        log_likelihood_old=log_likelihood
        
        for i in range(k):
            A[:,i]=likelihood(X,i,THETA)*c[i]
        log_likelihood=np.sum(np.log(A.sum(axis=1)))
        #print log_likelihood
        #print log_likelihood_old
        ###
        #for row in range(X.shape[0]):
        #    log_likelihood_Xi=0
        #    for i in range(k): # i th cluster
        #        p=likelihood(X[row],i,THETA)*c[i]
        #        A[row,i]=p
        #        log_likelihood_Xi+=p
        #    log_likelihood+=np.log(log_likelihood_Xi)
        A=A/A.sum(axis=1).reshape(m,1)
        #print log_likelihood
        #print log_likelihood_old
        
        # M-step:
        
        # 1. recompute weights for cluster
        c=A.sum(axis=0)/m
        #print c
        
        # 2. reassign parameters for each cluster
        for i in range(k):
            A_selected=A[:,i]    
            v1=(X==0).T.dot(A_selected)/A_selected.sum(axis=0)
            v2=(X==1).T.dot(A_selected)/A_selected.sum(axis=0)
            THETA[i][:,0]=v1
            THETA[i][:,1]=v2
        
    return [THETA,A,c,log_likelihood]

def select_stable_models(X,k):
    """
    
    This function will train 10 times for each chosen k.
    It will return the model which performs the best
    Performance is based on log likelihood
    
    """
    THETA,A,c,log_likelihood=train_cluster_model(X,k)
    for i in range(10):
        THETA_temp,A_temp,c_temp,log_likelihood_temp=train_cluster_model(X,k)
        #print (c_temp)
        if log_likelihood_temp>log_likelihood:
            THETA=THETA_temp
            A=A_temp
            c=c_temp
            log_likelihood=log_likelihood_temp
    return [THETA,A,c,log_likelihood]

def main():
    import os
    k=3 # set the number of clusters
    X=np.load("../output/train1_matrix.npy")
    THETA,A,c=train_cluster_model(X,k)
    filename='cluster_'+str(k)+'_model.npz'
    base_dir="../output"
    np.savez(os.path.join(base_dir,filename),THETA=THETA,A=A,c=c)

if __name__=="__main__":        
    main()
    
