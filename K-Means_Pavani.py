
# coding: utf-8

# In[64]:

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from pandas.plotting import andrews_curves


# In[101]:

#get random values for k centroids
def getCentroids(k):    
    x = np.random.uniform(0.0,1.0 , size=k)
    y = np.random.uniform(0.0,1.0 , size=k)
    C = np.array(list(zip(x, y)), dtype=np.float32)
    return C
def euclidean_dist(a,b):
    fin = np.empty((0,len(b)))
    for i in range(len(a)):        
        dist = (b - a[i])**2
        dist = np.sum(dist, axis=1)
        fin = np.vstack([fin, dist])
    return fin
def showPlot(k,data,columns,res, centroids):
    colors = ['b', 'y', 'c', 'm','r','g']
    fig, ax = plt.subplots()
    for i in range(k):
            points = data.loc[data[res] == i+1]
            ax.scatter(points[columns[0]], points[columns[1]], s=7, c=colors[i], label = i+1)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=30, c='#000000')
    ax.legend()
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.show()
def clusterSSE(points, centroid):
    dist = euclidean_dist(centroid.reshape(-1,len(centroid)),points)
    return dist.sum()
def calculateMeasures(k,data,res,columns, centroids):
    size = []
    total_SSE =0
    for i in range(k):
        points = data.loc[data[res] == i+1].as_matrix(columns = columns)
        size.append(len(points))
        x = clusterSSE(points,centroids[i,:])
        total_SSE+=x
        print("SSE for cluster ",i+1,"(",len(points),"points): ",x )

    main_centroid = data[columns].mean().values.reshape(-1,2)
    print("total SSE: ",total_SSE)
    SSB = 0
    for i in range(k):
        SSB +=  size[i] * euclidean_dist(main_centroid,centroids[i,:].reshape(-1,2))
    print("SSB: ",SSB[0][0])
    print("Total sum of squares = ",total_SSE+SSB[0][0])
def K_Means(k,data,columns,org, n_iter):
    #get centroid
    centroids = getCentroids(k)
    #normalising values
    #for feature_name in columns:
    #    max_value = data[feature_name].max()
    #    min_value = data[feature_name].min()
    #    data[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    data_mat = data.as_matrix(columns = columns)
    data['res'] = data[org]
    iterations =0;
    while True:
        dist_mat = euclidean_dist(centroids,data_mat)
        (row, col) = dist_mat.shape
        for i in range(col):
            x = dist_mat[:,i]
            data.set_value(i, 'res', np.where(x == np.min(x))[0][0] +1)
        new_centroids = data.groupby(['res'])[columns].mean().as_matrix(columns = columns)
        iterations = iterations+1
        if(np.array_equal(centroids, new_centroids) or iterations>n_iter):
            break;
        centroids = new_centroids
    showPlot(k,data,columns,'res',centroids)
    print("number of iterations : ",iterations)
    calculateMeasures(k,data,'res',columns,centroids)
    data[['ID','res']].to_csv("output_Pavani.csv", sep = ',',columns= ['ID','res'])
    print(data[['cluster','res','ID']].groupby(['cluster','res']).agg([len]))


# In[ ]:




# In[102]:

data = pd.read_csv("TwoDimHard.csv", delimiter='\,', engine = 'python')
data.describe()


# In[103]:

true_centroids = data.groupby(['cluster'])['X.1','X.2'].mean().as_matrix(columns = ["X.1","X.2"])
showPlot(4,data,['X.1','X.2'],'cluster',true_centroids) 
print("calculating true measures:")
calculateMeasures(4,data,'cluster',['X.1','X.2'],true_centroids)


# In[104]:

K_Means(3,data,["X.1","X.2"],'cluster',200)


# In[105]:

K_Means(4,data,["X.1","X.2"],'cluster',200)

