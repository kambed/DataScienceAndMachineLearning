from sklearn.cluster import KMeans
from enum import Enum
from clustering.clustering_algorithm import ClusteringAlgorithm
from matplotlib import pyplot as plt

"""
    k_means class
"""


def plot_elbow_curve(n_clusters_range, inertia_values):
    plt.plot(n_clusters_range, inertia_values, marker='o')
    plt.title('Elbow Curve')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


class KMeansAlgorithm(ClusteringAlgorithm):
    """
            :param n_clusters: Number of clusters
            :param init: Method for initialization centroids{k-means++, random}
        """

    def __init__(self, data, n_clusters, init):
        self.n_clusters = n_clusters
        self.init = init
        model = KMeans(n_clusters=n_clusters, init=init.value)
        super().__init__(data, model)

    def create_elbow_inertia_chart(self, n_clusters_range):
        inertia_values = []
        data = self.data.drop('Class', axis=1)
        for n in range(1, n_clusters_range + 1):
            kmeans = KMeans(n_clusters=n, init=self.init.value)
            kmeans.fit(data)
            inertia_values.append(kmeans.inertia_)

        plot_elbow_curve(range(1, n_clusters_range + 1), inertia_values)


class Init(Enum):
    RANDOM = 'random'
    KMEANS_PLUS_PLUS = 'k-means++'

