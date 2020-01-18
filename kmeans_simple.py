import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from util import get_data

df = get_data().sample(20000)
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(num_df)

X = num_df
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = StandardScaler().fit_transform(X)

y = pd.Categorical(df["race"]).codes
from util import plot2d
from util import plot3d
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

# plot2d(X, labels, labels, mode=PCA, centroids=kmeans.cluster_centers_)
# plot2d(X, kmeans.labels_, y, mode=PCA, centroids=kmeans.cluster_centers_)
# plot2d(X, y, y, mode=PCA)
# plot2d(X, labels, labels, mode=PCA, centroids=kmeans.cluster_centers_)
#
# plot3d(X, y, y, mode=PCA)
# plot3d(X, labels, labels, mode=PCA, centroids=kmeans.cluster_centers_)

for x in cat_df_list:
    y = pd.Categorical(df[x]).codes
    plot3d(X, y, y, mode=PCA, name=x)

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
