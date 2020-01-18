import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from util import get_data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier


df = get_data()
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

model = KMeans(n_clusters=5)
labels = model.fit_predict(num_df)

X = num_df
X_train, X_test, y_train, y_test = train_test_split(X, df["gender"], test_size=0.2, random_state=9)
from sklearn.model_selection import train_test_split

# xgboost Classifier with grid search

model = xgb()
grid_param = {
    "learning_rate": [0.01, 0.1],
    'max_depth': [15, 25, 50],
    'n_estimators': [10, 100, 200, 1000]
}
grid = GridSearchCV(model, grid_param, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
# print results
grid.cv_results_
