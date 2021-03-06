import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from munkres import Munkres, print_matrix

from util import get_data
from util import plot2d
from util import plot3d
from mpl_toolkits import mplot3d

df = get_data()
df["gender_t"] = df["gender"].map({"Female": 0, "Male": 1})
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

from sklearn.metrics import cluster
from sklearn.cluster import KMeans

# m = Munkres()
# indexes = m.compute(matrix)
# print_matrix(matrix, msg='Lowest cost through this matrix:')
# total = 0
# for row, column in indexes:
#     value = matrix[row][column]
#     total += value
#     print(f'({row}, {column}) -> {value}')
# print('total cost: total', total)
from sklearn.manifold import TSNE
X = df[num_df_list]
clusterer = KMeans(n_clusters=2, random_state=10)
cluster_labels = clusterer.fit_predict(df[num_df_list])

silhouette_avg = metrics.silhouette_score(X, cluster_labels)
print("For n_clusters =", 2,
      "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)
