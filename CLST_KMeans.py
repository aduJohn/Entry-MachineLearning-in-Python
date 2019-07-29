# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:57:25 2019

@author: Alexandru
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:,-2:].values

#using the elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,color='red')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

#using and seeing the KMeans algorithm for 5 clusters
kmeans = KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_means = kmeans.fit_predict(X)

plt.scatter(X[y_means==0,0],X[y_means==0,1],s=100,color='red',label='Cluster 1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,color='blue',label='Cluster 2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=100,color='green',label='Cluster 3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=100,color='cyan',label='Cluster 4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=100,color='magenta',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,color='yellow',label='Centroids')
plt.ylabel('Spending score')
plt.xlabel('Annual Income')
plt.title('Clusttering the clients using KMeans')
plt.show()