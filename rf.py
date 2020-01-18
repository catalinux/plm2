from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
rf = RandomForestClassifier(n_estimators = 10, max_depth=25)
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
# create the RFE model for the svm classifier
# and select attributes
rfe = RFE(rf, 15)
X_new = rfe.fit_transform(os_data_X, os_data_y)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)