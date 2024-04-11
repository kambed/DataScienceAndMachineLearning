from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

"""
    ClusteringEvaluator class
"""


class ClusteringEvaluator:

    @staticmethod
    def evaluate(labels, X):
        db_score = davies_bouldin_score(X, labels)
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        print(f'Davies Bouldin Score: {db_score}')
        print(f'Silhouette Score: {silhouette}')
        print(f'Calinski Harabasz Score: {calinski}')
