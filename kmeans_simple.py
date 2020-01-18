import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from util import get_data

df = get_data()
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

model = KMeans(n_clusters=5)
labels = model.fit_predict(num_df)

X = num_df
y= df["gender"]
from util import plot2d
from sklearn.manifold import TSNE

plot2d(X, labels, labels, mode=TSNE)

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
