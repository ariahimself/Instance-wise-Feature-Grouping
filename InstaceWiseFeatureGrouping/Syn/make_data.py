# from __future__ import print_function
import numpy as np  
from scipy.stats import chi2

def generate_XOR_labels(X):


    y =np.exp(X[:,0]*X[:,2])


    prob_1 = np.expand_dims(1 / (1+y) ,1)
    prob_0 = np.expand_dims(y / (1+y) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:,:4]**2, axis = 1) - 4.0) 

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_additive_labels(X):
    logit = np.exp( np.sin(X[:,0]) + abs(X[:,1]) + X[:,2] + np.exp(-X[:,3])  - 2.4) 

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y



def generate_data(n=100, datatype='', seed = 0, val = False):
    """
    Generate data (X,y)
    Args:
        n(int): number of samples 
        datatype(string): The type of data 
        choices: 'orange_skin', 'XOR', 'regression'.
        seed: random seed used
    Return: 
        X(float): [n,d].  
        y(float): n dimensional array. 
    """

    np.random.seed(seed)


    X = np.random.randn(int(n), 10)

    mean = [0, 0,0,0]
    cov = [[1,0,0.9,0], 
            [0,1,0,0.9],
            [0.9,0,1,0],
            [0,0.9,0,1]]


    mean = [0, 0,0,0]
    cov = [[1,0,1,0], 
            [0,1,0,1],
            [1,0,1,0],
            [0,1,0,1]]


    X2 = np.random.multivariate_normal(mean, cov, int(n))


    mean3 = [0, 0,0,0]
    cov3 = [[1,0,0.9,0.9], 
            [0,1,0,0],
            [0.9,0,1,0.9],
            [0.9,0,0.9,1]]

    mean3 = [0, 0,0,0]
    cov3 = [[1,0,1,1], 
            [0,1,0,0],
            [1,0,1,1],
            [1,0,1,1]]



    X3 = np.random.multivariate_normal(mean3, cov3, int(n))







    mean1 = [0, 0,0,0]
    cov1 = [[1,0.9,0.0,0.0],
            [0.9, 1,0.0,0.0],
            [0.0,0.0,1,0.9],
            [0.0,0.0,0.9,1]] 


    mean1 = [0, 0,0,0]
    cov1 = [[1,1,0.0,0.0],
            [1, 1,0.0,0.0],
            [0.0,0.0,1,1],
            [0.0,0.0,1,1]] 

    X1 = np.random.multivariate_normal(mean1, cov1, int(n))
    #X= np.concatenate((X1, X2), axis=0)
    #X = np.random.randn(n, 10)
    #X4 = X3[int(n/2):,:]
    #X= np.concatenate((X1, X2), axis=0)
    datatypes = None
    datatypes = np.array(['orange_skin'] * len(X1) + ['nonlinear_additive'] * len(X2)) 

    X[:,0:4] = X2[:,0:4]
    # X[:int(n/2),0:4] = X1[:,0:4]
    # X[int(n/2):,0:4] = X2[:,0:4]11
    #X = np.random.randn(n, 4)
    #X = X1
     

    if datatype == 'orange_skin': 
        y = generate_orange_labels(X) 

        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]

    elif datatype == 'XOR':
        y= generate_XOR_labels(X) 

        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]


    elif datatype == 'nonlinear_additive':  
        y = generate_additive_labels(X) 

        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]

    elif datatype == 'switch':

        # Construct X as a mixture of two Gaussians.
        #X[:n//2,-1] += 3
        #X[n//2:,-1] += -3
        #X1 = X[:n//2]; X2 = X[n//2:]

        y1 = generate_orange_labels(X1)
        y2 = generate_additive_labels(X2)

        # Set the key features of X2 to be the 4-8th features.
        #X2[:,4:8],X2[:,:4] = X2[:,:4],X2[:,4:8]

        X = np.concatenate([X1,X2], axis = 0)
        y = np.concatenate([y1,y2], axis = 0) 

        # Used for evaluation purposes.
        datatypes = np.array(['orange_skin'] * len(y1) + ['nonlinear_additive'] * len(y2)) 

        # Permute the instances randomly.
        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]


    return X, y, datatypes  