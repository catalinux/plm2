import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from munkres import Munkres, print_matrix

from util import get_data

df = get_data()

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

from sklearn.metrics import cluster
from sklearn.cluster import KMeans

x = []
rng = range(2, 14)
for i in rng:
    kmeans = KMeans(n_clusters=i, random_state=0).fit(num_df)
    x.append( kmeans.inertia_)

plt.plot(rng, x, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
# matrix = cluster.contingency_matrix(df['readmitted'], kmeans.labels_)

# m = Munkres()
# indexes = m.compute(matrix)
# print_matrix(matrix, msg='Lowest cost through this matrix:')
# total = 0
# for row, column in indexes:
#     value = matrix[row][column]
#     total += value
#     print(f'({row}, {column}) -> {value}')
# print('total cost: total', total)
