import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from munkres import Munkres, print_matrix

df = pd.read_csv('dataset_diabetes/diabetic_data.csv', na_values='?')
df2 = df.drop_duplicates(subset=['patient_nbr'], keep='first')

df = df.drop(["weight", "payer_code", "medical_specialty"], axis=1)
df = df.drop(df[df.gender == 'Unknown/Invalid'].index)
meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
        'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
df = df.drop(meds, axis=1)

df['readmitted'] = df['readmitted'].replace('>30', 0)
df['readmitted'] = df['readmitted'].replace('<30', 1)
df['readmitted'] = df['readmitted'].replace('NO', 0)
df['service_utilization'] = df['number_outpatient'] +    df['number_emergency'] + df['number_inpatient']
age_dict = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
            "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
df['age'] = df.age.map(age_dict)
df['age'] = df['age'].astype('int64')

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))

from sklearn.metrics import cluster
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(df[num_df_list])
matrix = cluster.contingency_matrix(df['readmitted'], kmeans.labels_)

m = Munkres()
indexes = m.compute(matrix)
print_matrix(matrix, msg='Lowest cost through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print('total cost: total', total)
