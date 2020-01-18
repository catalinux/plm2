from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

from util import get_data
from util import prepare_data
from util import plot2d
import pandas as pd
df = get_data()
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
X = df[num_df_list]
y = df['readmitted']

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

y = df["readmitted"]
X = prepare_data(df)
X.drop("readmitted", inplace=True, axis=1)
scaler = StandardScaler()
X = StandardScaler().fit_transform(X)

from imblearn.under_sampling import (RandomUnderSampler,
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

sampler = NearMiss(n_jobs=2)
X_rs, y_rs = sampler.fit_sample(X, y)


# sc=[]
# for i in np.linspace(3,40):
#     clustering = DBSCAN(eps=i, min_samples=2).fit(X)
#     labels=clustering.labels_
#
#     clustering
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     print(n_clusters_)
#     sc.append({
#         "clusters":n_clusters_,
#         "noise":n_noise_,
#         "eps":i
#     })


# plot2d(X, clustering.labels_, y, mode=PCA)


# Init NearestNeighbors with n_neighbors = 4 and fit on the training data
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(X)

# Find the nearest neighbors of the data points and determine the distances
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]

plt.plot(distances)
plt.show()
# fig = plt.figure()
# l = [2, 3,40,60]
# for x in l:
#     # Init NearestNeighbors with n_neighbors = 4 and fit on the training data
#     neigh = NearestNeighbors(n_neighbors=x)
#     nbrs = neigh.fit(X)
#     # Find the nearest neighbors of the data points and determine the distances
#     distances, indices = nbrs.kneighbors(X)
#     distances = np.sort(distances, axis=0)
#     distances = distances[:, 1]
#     plt.plot(distances)
#
# fig.savefig('img/epsilon.png')
# plt.legend(l)
# plt.show()

m = DBSCAN(eps=3, min_samples=5)
m.fit(X)
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan',
          'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

#
# clusters = m.labels_
# plt.scatter(X["time_in_hospital"], X["num_lab_procedures"], c=vectorizer(clusters))
#
# #plot2d(X, clustering.labels_, y, mode=PCA)
#
# plt.figure()
#
# from sklearn.manifold import TSNE
# plot2d(X, m.labels_, y, TSNE)
#
# plt.figure()
# #plot2d(X, y, y, TSNE)
