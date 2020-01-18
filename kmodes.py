from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

from util import get_data
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
X = df[cat_df_list]

import numpy as np
from kmodes.K import KModes

# random categorical data
data = np.random.choice(20, (100, 10))

km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

# Print the cluster centroids
print(km.cluster_centroids_)