import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import plot2d

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from util import get_data
from util import get_unclean_data

df = get_data()
cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


def explaind_variance():
    global num_df
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


explaind_variance()

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


def heatmap():
    corr = pd.DataFrame(num_df).corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


# heatmap()


def plot_categorical():
    for x in cat_df_list:
        plt.figure()
        sns.countplot(x=x, data=df)
        plt.show()


# plot_categorical()


def histnum():
    for x in num_df_list:
        plt.figure()
        plt.hist(num_df[x])
        plt.title(x)
        plt.show()

# histnum()
