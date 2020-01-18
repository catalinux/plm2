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
X = df[num_df_list]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = StandardScaler().fit_transform(X)

diabetic_data = X
db_scan = DBSCAN(eps=3, min_samples=4).fit(diabetic_data)

labels = db_scan.labels_
