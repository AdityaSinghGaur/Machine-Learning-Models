# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:11:08 2020

@author: Aditya Singh Gaur
"""

#importing the librarires
import pandas as pd
import matplotlib.pyplot as plt
import numpy as nm

#reading the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: , [3,4]].values

#using the dendograms to find the number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X , method ='ward'))
plt.title('Dendograms')
plt.xlabel('Clusters')
plt.ylabel('Euclidean Distances')
plt.show()

#fitting hierarchial clustering into the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean' , linkage = 'ward')
y_hc = hc.fit_predict(X)

#vizualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Cluster 1' )
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Cluster 2' )
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Cluster 3' )
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Cluster 4' )
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='Cluster 5' )
plt.title('Clusters of Clusters')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spemding Score(1-100)')
plt.legend()
plt.show()