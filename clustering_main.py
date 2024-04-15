from clustering.agglomerate_clustering import AgglomerateClustering, Linkage
from clustering.clustering_algorithm import Clustering
from clustering.clustering_evaluator import ClusteringEvaluator
from clustering.dbscan import Dbscan
from helper.argument_helper import ArgumentHelper
from helper.data_csv_helper import DataCsvHelper


def create_clustering_algorithm(data):
    algorithm = ArgumentHelper.get_enum_argument("algorithm", Clustering)
    if algorithm == Clustering.AGGLOMERATIVE:
        n_clusters = ArgumentHelper.get_int_argument("n_clusters")
        linkage = ArgumentHelper.get_enum_argument("linkage", Linkage)
        return AgglomerateClustering(data, n_clusters, linkage)
    elif algorithm == Clustering.DBSCAN:
        eps = ArgumentHelper.get_float_argument("eps")
        min_samples = ArgumentHelper.get_int_argument("min_samples")
        return Dbscan(data, eps, min_samples)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not found!")


if __name__ == '__main__':
    labels = ['DERMASON', 'SIRA', 'SEKER']
    data = DataCsvHelper.read_csv(labels=labels)

    clustering = create_clustering_algorithm(data)
    algorithms_labels, X, y = clustering.fit()
    ClusteringEvaluator.evaluate(algorithms_labels, X, y)
