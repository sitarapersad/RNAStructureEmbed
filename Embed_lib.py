import numpy as np
import pandas as pd
import logging as log
import plotly
from plotly.offline import plot
plotly.plotly.sign_in('spersad', 'oNkuP1yzbpN734Ag8M9P')
import plotly.graph_objs as go

import scipy

log.getLogger().setLevel(log.INFO)

def mask_dependent_bases(X):
    '''
    Given a bitvector, replace all bases around '1' with np.nan 
    '''
    print(X.dtype)
    X_pad = np.lib.pad(X.astype(np.float64), (3,3), 'constant', constant_values=(0.0, 0.0))[3:-3]
    X_pad = X_pad.reshape(1,-1)[0]
    ones = np.where(X_pad==1)[0]
    X_pad[ones-1] = np.nan
    X_pad[ones-2] = np.nan
    X_pad[ones-3] = np.nan
    
    X_pad[ones+1] = np.nan
    X_pad[ones+2] = np.nan
    X_pad[ones+3] = np.nan
    X_masked = X_pad.reshape(-1,X.shape[1]+6)
    
    return X_masked


def first_order_dist(X):
    '''
    Compute the Hamming distance between reads, where dist(i,i)=dist(nan,i)=0 and dist(i,j)=1
    @param: X -  numpy array of bitvectors 
    @return: dist - distance matrix which is the first-order Hamming distance between reads
    '''
    dist = np.zeros((X.shape[0],X.shape[0]))
    X_no_nan = np.nan_to_num(X)
    print('Computing first order distance matrix with shape {0}'.format(dist.shape))
    # for i in range(len(X)):
    #     for j in range(i, len(X)):
    #         row1 = X[i]
    #         row2 = X[j]
    #         d = np.nansum(np.abs(row1-row2))
    #         dist[i,j] = d
    #         dist[j,i] = d
    # return dist
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X_no_nan, metric='hamming'))

def second_order_dist(X, k=5):
    ''' 
    Given a set of reads, compute the second order distance matrix,
    where the distance between reads i, j is the d(i,j) + alpha Sum(d(i,k_j)+d(k_i,j))
    and k_j is the k-NN neighborhood of j and k_i is the k-NN neighborhood of i
    @param: X -  numpy array of bitvectors
    @param: k - number of neighbors to consider 
    @return: second_dist - distance matrix which is the second-order Hamming distance between reads
    '''

    ALPHA=1
    def mydist(x,y):
        return np.nansum(np.abs(x-y))
    
    from sklearn.neighbors import NearestNeighbors as NN
    nbrs  = NN().fit(X) #(algorithm='ball_tree', metric=mydist
    dists, indices = nbrs.kneighbors(n_neighbors=k)
    print('Computing second order distance matrix with shape {0}'.format(dists.shape))
    indices = indices[:,1:k] # Each read is not in its own nearest neighborhood
    
    second_dist = np.zeros((X.shape[0],X.shape[0]))
    print(second_dist.shape)
    for i in range(len(X)):
        for j in range(i, len(X)):
            row1 = X[i]
            row2 = X[j]
            d = np.nansum(np.abs(X[i]-X[j])) + ALPHA*(np.nansum(np.abs(X[indices[i]] - X[j])) + np.nansum(np.abs(X[indices[j]] - X[i])))
            second_dist[i,j] = d
            second_dist[j,i] = d
    return second_dist

def performMDS(X, Y, weight=None, text=None,num_examples=2000,metric='precomputed', outDir=None):
    '''
    @param: weight - multiplier which scales point sizes
    @param: text - hover text for points
    '''
    from sklearn.manifold import MDS
    model = MDS(n_components=2, max_iter=3000, dissimilarity=metric)
    print('Defined MDS model')
    if metric !='precomputed':
        X = X[:num_examples]
        Y = Y[:num_examples]
    print('Performing MDS on data with shape {0}'.format(X.shape))         
    embeddedX = model.fit_transform(X)
    print('Created embedding')
    if text is None:
        text = [str(x) for x in X]
    if weight is None:
        weight = np.array([0.5 for x in X])
    # Scatter plot to visualize embedded data
    # Create a trace
    trace = go.Scatter(
        x = embeddedX[:,0],
        y = embeddedX[:,1],
        mode = 'markers',
        marker=dict(
            size=10*weight,
            color = Y, # color points by label they belong to
            colorscale= [[0, '#dd2c4f'], [1, '#3d6fcc']],
        ),
        text = text,
    )

    data = [trace]
    
    layout = go.Layout(
        title='MDS Embedding of Clusters in 2D Space',
    )

    fig = go.Figure(data=data, layout=layout)

    if outDir != None:
        plot(fig, auto_open=False, filename=outDir+'/MDS_embedding.html')
        print('Saved plot to {0}'.format(outDir))
    return plot(fig, auto_open=False, output_type='div')

def performTSNE(X, Y, weight=None, text=None, num_examples=2000, perp=30, metric='precomputed', outDir=None):
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, perplexity = perp, random_state=0, metric=metric) # fit into 2D space
    print('Defined tSNE model')
    if metric !='precomputed':
        X = X[:num_examples]
        Y = Y[:num_examples]
    print('Performing TSNE on data with shape {0}'.format(X.shape))        
    embeddedX = model.fit_transform(X)
    print('Created embedding')
    if text is None:
        text = [str(x) for x in X]
    if weight is None:
        weight = np.array([0.5 for x in X])

    # Scatter plot to visualize embedded data
    # Create a trace
    trace = go.Scatter(
        x = embeddedX[:,0],
        y = embeddedX[:,1],
        mode = 'markers',
        marker=dict(
            size=10*weight,
            color = Y, # color points by label they belong to
            colorscale= [[0, '#dd2c4f'], [1, '#3d6fcc']],
        ),
        text = text,
    )

    data = [trace]
    
    layout = go.Layout(
        title='Embedding of Clusters in 2D Space',
    )

    fig = go.Figure(data=data, layout=layout)
    if outDir != None:
        plot(fig, auto_open=False, filename=outDir+'/tSNE_embedding.html')
        print('Saved plot to {0}'.format(outDir))
    return plot(fig, auto_open=False, output_type='div')
    