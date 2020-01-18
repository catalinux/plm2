from imblearn.under_sampling import (RandomUnderSampler,
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



titles = {
    "km_scores": "The Elbow Method",
    'km_silhouette': "Silhouette - The best value is 1 and the worst value is -1",
    'db_score': "The minimum score is zero, with lower values indicating better clustering",
    'vmeasure_score': "",
    "inertia": "The Elbow Method."
}

labels = {
    "km_scores": "Inertia",
    'km_silhouette': "Silhouette",
    'db_score': "Davies-Bouldin score",
    'vmeasure_score': "",
    "inertia": "Distortion value"
}

a = pd.read_pickle('score.bin2')
list = {
    "km_scores": "",
    'km_silhouette': "",
    'db_score': "",
    'vmeasure_score': "",
    "inertia": ""
}
for col in ['km_scores',  'db_score']:
    fig = plt.figure()
    list[col] = (fig)
    plt.plot(a["k"], a[col], 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel(labels[col])
    plt.title(titles[col])
    plt.show()
    png = './img/km_scores_50' + col + '.png'
    print("![" + col + "](" + png + "){width=33%} ")
    fig.savefig(png)
