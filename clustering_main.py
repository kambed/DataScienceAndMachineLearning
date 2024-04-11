from clustering.agglomerate_clustering import AgglomerateClustering
from clustering.clustering_algorithm import Clustering
from clustering.clustering_evaluator import ClusteringEvaluator
from helper.argument_helper import ArgumentHelper
from helper.data_csv_helper import DataCsvHelper


def create_clustering_algorithm(data):
    algorithm = ArgumentHelper.get_enum_argument("algorithm", Clustering)
    if algorithm == Clustering.AGGLOMERATIVE:
        n_clusters = ArgumentHelper.get_int_argument("n_clusters")
        return AgglomerateClustering(data, n_clusters)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not found!")


if __name__ == '__main__':
    labels = ['DERMASON', 'SIRA', 'SEKER']
    data = DataCsvHelper.read_csv(labels=labels)

    clustering = create_clustering_algorithm(data)
    algorithms_labels, X = clustering.fit()
    ClusteringEvaluator.evaluate(algorithms_labels, X)
