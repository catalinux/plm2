from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import plot2d
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from util import get_data
from util import prepare_data
from util import get_unclean_data
from util import plot3d
from util import plot2d
from sklearn.decomposition import PCA

df = get_data()
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

# y = df["readmitted"]
# X = prepare_data(df)
# X.drop("readmitted", inplace=True, axis=1)
# scaler = StandardScaler()
# X = StandardScaler().fit_transform(X)
#
# from imblearn.under_sampling import (RandomUnderSampler,
#                                      ClusterCentroids,
#                                      TomekLinks,
#                                      NeighbourhoodCleaningRule,
#                                      NearMiss)
#
# sampler = NearMiss()
# X_rs, y_rs = sampler.fit_sample(X, y)
#
# tran = PCA(n_components=3)
# X_2_ = tran.fit_transform(X_rs)
#
# kmeans = KMeans(n_clusters=2)
# labels = kmeans.fit_predict(X_2_)
# plot3d(X_2_, y_rs, y_rs, TSNE)
# plot3d(X_2_, labels, labels, TSNE)

### only on numeric

y = df["readmitted"]
y_gender = df["gender"]
X = df[num_df_list]
scaler = StandardScaler()
X = StandardScaler().fit_transform(X)

from imblearn.under_sampling import (RandomUnderSampler,
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

# sampler = RandomUnderSampler()
# X_rs, y_rs = sampler.fit_sample(X, y)
#
# acc = []
# for i in range(3, 20):
#     tran = PCA(n_components=4)
#     X_2_ = tran.fit_transform(X_rs)
#     kmeans = KMeans(n_clusters=2)
#     labels = kmeans.fit_predict(X_2_)
#     mean = (labels == y_rs).mean()
#     print(mean)
#     acc.append(mean)
#
# plt.plot(acc)
# plt.show()
#
# # tran = PCA(n_components=2)
# # X_2_ = tran.fit_transform(X)
# kmeans = KMeans(n_clusters=2)
# labels = kmeans.fit_predict(X)
acc = []
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

for i in range(2, 13):
    tran = PCA(n_components=i)
    X_2_ = tran.fit_transform(X)
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(X_2_)
    mean = (labels == y).mean()
    print(i)
    print("Accuracy", accuracy_score(y, labels))
    c = confusion_matrix(y, labels)
    print(c)
    acc.append(mean)
#
# plt.plot(acc)
# plt.show()
#
# kmeans = KMeans(n_clusters=2)
# labels = kmeans.fit_predict(X)
# mean = (labels == y).mean()
# print(mean)
