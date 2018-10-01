# Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# load the stock data
stock_data = pd.read_csv('/home/john/Dropbox/Trident/Data/kaggle_data/marketdata_sample.csv')
sd_matrix = stock_data[['volume','close']].copy().values

sdm_top = sd_matrix[0:50,]
sdm_bot = sd_matrix[51:,]

# perform K analysis to figure out number of clusters
distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(sd_matrix)
    kmeanModel.fit(sd_matrix)
    distortions.append(sum(np.min(cdist(sd_matrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / sd_matrix.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# cluster the data and assign
kmeans = KMeans(n_clusters = 8)
kmeans.fit(sd_matrix)
stock_data['cluster'] = kmeans.labels_
kmeans.cluster_centers_
kmeans.predict(sd_matrix)

# Test grouping with and in and out sample
kmTest = KMeans(n_clusters = 8)
kmTest.fit(sdm_top)
kmTest.cluster_centers_
kmTest.labels_
kmeans.predict(sdm_bot)


# plot clusters, whole sample
X = sd_matrix
plt.scatter(X[:,0], X[:,1], c = kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color = 'black', marker = 'x')

# plot clusters, in sample
plt.scatter(sdm_top[:,0], sdm_top[:,1], c = kmTest.labels_, cmap='rainbow')
plt.scatter(kmTest.cluster_centers_[:,0] ,kmTest.cluster_centers_[:,1], color = 'black', marker = 'x')