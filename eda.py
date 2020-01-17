import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import plot2d

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('dataset_diabetes/diabetic_data.csv', na_values='?')
df.head()
df_missing = df.copy()
missing = df_missing.isnull().sum()
print(missing)
missing_rates = np.round(100 * missing[missing > 0] / df.__len__(), 3).to_clipboard()

df = df.drop(["weight", "payer_code", "medical_specialty", 'encounter_id', 'patient_nbr', 'admission_type_id',
              'discharge_disposition_id', 'admission_source_id'], axis=1)

df = df.drop(df[df.gender == 'Unknown/Invalid'].index)
meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
        'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
df = df.drop(meds, axis=1)
df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
age_dict = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
            "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
df['age'] = df.age.map(age_dict)
df['age'] = df['age'].astype('int64')
df=df.dropna()

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))

# for var in num_df_list:
#     sns_plot = sns.boxplot(x=var, y='PRICE', data=df)
#     sns_plot.set_title(var)
#     title = 'img/cat_bivar_price_' + var + '.png'
#     sns_plot.figure.savefig(title)
#     # for label in sns_plot.get_xticklabels():
#     #     label.set_rotation(45)
#     print("![" + var + "](./" + title + "){width=50%}")

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

pca = PCA(n_components=10)
num_df = df[num_df_list]

std = StandardScaler()
df_std = std.fit_transform(num_df)
num_df = std.transform(df_std)
principalComponents = pca.fit_transform(num_df)

sns.lineplot(data=np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("95% of  variance is explained by about 6 components")
plt.show()

# df[df['readmitted']=='NO'].plot(kind='scatter',x=num_df_list[0],y=num_df_list[1],color='r')
# df[df['readmitted']!='NO'].plot(kind='scatter',x=num_df_list[0],y=num_df_list[1],color='b')

import itertools


def generate_scatters():
    for a in set(list(itertools.combinations(num_df_list, 2))):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(a[0], a[1], a)
        ax1.scatter(df[df['gender'] == 'Female'][a[0]], df[df['gender'] == 'Female'][a[1]], s=10, c='r', marker="o",
                    label='Female', alpha=0.3)
        ax1.scatter(df[df['gender'] != 'Male'][a[0]], df[df['gender'] != 'Male'][a[1]], s=10, c='b', marker="o",
                    label='Male', alpha=0.3)
        ax1.set_ylabel(a[1])
        ax1.set_xlabel(a[0])
        ax1.legend()
    plt.show()


#generate_scatters()

from sklearn.manifold import TSNE
#plot2d(num_df, df["gender"],  df["gender"], TSNE)

# for a in set(list(itertools.combinations(num_df_list, 2))):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     print(a[0], a[1], a)
#     ax1.scatter(df[df['readmitted'] == 'NO'][a[0]], df[df['readmitted'] == 'NO'][a[1]], s=10, c='r', marker="o",
#                 label='NO', alpha=0.3)
#     ax1.scatter(df[df['readmitted'] != 'NO'][a[0]], df[df['readmitted'] != 'NO'][a[1]], s=10, c='b', marker="o",
#                 label='YES', alpha=0.3)
#     ax1.set_ylabel(a[1])
#     ax1.set_xlabel(a[0])
#     ax1.legend()
#
# plt.show()
#


corr = pd.DataFrame(num_df).corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,  cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
