import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from munkres import Munkres, print_matrix

from sklearn import metrics

from sklearn.metrics import cluster
from sklearn.cluster import KMeans

from util import get_data

df = get_data()

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
km_scores = []
inertia = []
km_silhouette = []
vmeasure_score = []
db_score = []

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numdf = df[num_df_list]
X_scaled = scaler.fit_transform(numdf)
y = df["gender"]
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score

r = range(2, 12)
for i in r:
    km = KMeans(n_clusters=i, random_state=0, n_jobs=64).fit(X_scaled)
    preds = km.predict(X_scaled)

    print("Score for number of cluster(s) {}: {}".format(i, km.score(X_scaled)))
    km_scores.append(-km.score(X_scaled))

    inertia.append(km.inertia_)
    silhouette = silhouette_score(X_scaled, preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i, silhouette))

    db = davies_bouldin_score(X_scaled, preds)
    db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i, db))



    # v_measure = v_measure_score(y, preds)
    # vmeasure_score.append(v_measure)
    # print("V-measure score for number of cluster(s) {}: {}".format(i, v_measure))
    # print("-" * 100)

scores = pd.DataFrame.from_dict({
    "k": r,
    "km_scores": km_scores,
    "km_silhouette": km_silhouette,
    "db_score": db_score,
    "inertia": inertia,
#    "vmeasure_score": vmeasure_score,
})

with open('score.bin', 'wb') as fp:
    pickle.dump(scores, fp)
