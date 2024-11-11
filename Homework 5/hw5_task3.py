#
# Template for Task 3: Kmeans Clustering
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
# note we do not need label 
data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
sample = data[:,0:-1]
# -------------------------------------

# --- Your Task --- #
# pick a proper number of clusters 
k = 3
# --- end of task --- #


# --- Your Task --- #
# implement the Kmeans clustering algorithm 
# you need to first randomly initialize k cluster centers 
centers = sample[np.random.choice(n, k, replace=False)]

# intialize all labels to 0 cluster
label_cluster = np.zeros(n, dtype=int)
# then start a loop 

converged = False
while(not converged):

     for i in range(n):
          distances = np.linalg.norm(sample[i] - centers, axis=1)  # Distance to each cluster center
          label_cluster[i] = np.argmin(distances)  # Assign point to nearest cluster
    
     new_centers = np.zeros_like(centers)
     for j in range(k):
          points_in_cluster = sample[label_cluster == j]
          if len(points_in_cluster) > 0:  # Avoid division by zero
               new_centers[j] = points_in_cluster.mean(axis=0)
    
     # Check for convergence (if centers do not change)
     if np.all(centers == new_centers):
          converged = True
     else: 
          centers = new_centers

# --- end of task --- #


# the following code plot your clustering result in a 2D space
pca = PCA(n_components=3)
pca.fit(sample)
sample_pca = pca.transform(sample)
idx = []
colors = ['blue','red','green','m']
for i in range(k):
     idx = np.where(label_cluster == i)
     plt.scatter(sample_pca[idx,0],sample_pca[idx,1],color=colors[i],facecolors='none')
plt.show()