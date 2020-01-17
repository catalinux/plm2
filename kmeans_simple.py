import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from data_inference.dimensions.dimensional_reducer import DimensionalReducer

KMEANS_NUM_ITERATIONS = 20


def apply_kmeans_clustering(data_set, dimensions, clusters):
    diabetic_data = pd.read_csv(data_set)
    diabetic_data = DimensionalReducer(diabetic_data).reduce_dimension(dimensions)
    model = KMeans(n_clusters=clusters,
                   n_init=KMEANS_NUM_ITERATIONS)
    labels = model.fit_predict(diabetic_data)

    # Assign the cluster centers: centroids
    centroids = model.cluster_centers_
    print(centroids)

    plt.scatter(diabetic_data[:, 0], diabetic_data[:, 1], diabetic_data[:, 2], c=labels)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(diabetic_data[:, 0], diabetic_data[:, 1], diabetic_data[:, 2], c=labels)
    plt.show()
