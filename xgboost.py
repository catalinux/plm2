from mpl_toolkits.mplot3d import Axes3D
from util import get_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import time

df = get_data()
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df = df[num_df_list]

df_attr = num_df
df_target = df["gender"]
attr_train, attr_test, target_train, target_test = train_test_split(df_attr, df_target)

models = [('Decision Tree', DecisionTreeClassifier(max_depth=5), 'red'),
          ('Logistic Regression', LogisticRegression(n_jobs=30), 'green'),
          ('Random Forest', RandomForestClassifier(n_jobs=30), 'yellow'),
          ('Adaboost', AdaBoostClassifier(), 'magenta'),
          ('Neural Network', MLPClassifier(hidden_layer_sizes=4, max_iter=10000), 'blue')]

predicted_results = {}
time_elapsed = []
cross_val_list = []
for model_name, model, _ in models:
    start = time.time()
    model.fit(attr_train, target_train)
    target_predict = model.predict(attr_test)

    score = cross_val_score(model, df_attr, df_target, cv=5)
    cross_val_list.append(score)

    end = time.time()

    time_elapsed.append(end - start)
    predicted_results[model_name] = target_predict
