import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from util import get_data

df = get_data()
df["gender_t"] = df["gender"].map({"Female": 0, "Male": 1})
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

model = KMeans(n_clusters=4)
labels = model.fit_predict(num_df)

X = num_df

from util import plot2d
from sklearn.manifold import TSNE
#plot2d(X, labels, df["gender_t"],mode=TSNE)
