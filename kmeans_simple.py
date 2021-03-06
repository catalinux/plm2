import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from util import get_data
from util import prepare_data

df = get_data()
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

from sklearn.preprocessing import StandardScaler

X = num_df

X = df[num_df_list]
y = df["readmitted"]
# X.drop(["readmitted"],axis=1, inplace=True)

# scaler = StandardScaler()
# X = StandardScaler().fit_transform(X)

from imblearn.under_sampling import (RandomUnderSampler,
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

sampler = NearMiss()
X_rs, y_rs = sampler.fit_sample(X, y)

from util import plot2d
from util import plot3d
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

# plot2d(X_rs, labels, labels, mode=TSNE)
kmeans = KMeans(n_clusters=8)
labels = kmeans.fit_predict(X_rs)
plot2d(X_rs, labels, labels, mode=TSNE)
plot3d(X_rs, labels, labels, mode=TSNE)

kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(X_rs)
plot2d(X_rs, labels, y_rs, mode=TSNE)

plot2d(X_rs, labels, labels, mode=TSNE, centroids=kmeans.cluster_centers_)

# plot2d(X, labels, labels, mode=PCA, centroids=kmeans.cluster_centers_)
# plot2d(X, kmeans.labels_, y, mode=PCA, centroids=kmeans.cluster_centers_)
# plot2d(X, y, y, mode=PCA)
# plot2d(X, labels, labels, mode=PCA, centroids=kmeans.cluster_centers_)
#
# plot3d(X, y, y, mode=PCA)
# plot3d(X, labels, labels, mode=PCA, centroids=kmeans.cluster_centers_)
#
# for x in cat_df_list:
#     y = pd.Categorical(df[x]).codes
#     plot3d(X, y, y, mode=PCA, name=x)

# count_class_0, count_class_1 = df["readmitted"].value_counts()
#
# # Divide by class
# df_class_0 = df[df['readmitted'] == 0]
# df_class_1 = df[df['readmitted'] == 1]
#
# df_class_0_under = df_class_0.sample(count_class_1)
# new_df = pd.concat([df_class_0_under, df_class_1], axis=0)
#
# y = new_df["readmitted"]
# plot2d(new_df[num_df_list], y, y, mode=TSNE)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
