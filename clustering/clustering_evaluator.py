from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, \
    rand_score, fowlkes_mallows_score, v_measure_score

"""
    ClusteringEvaluator class
"""


class ClusteringEvaluator:

    @staticmethod
    def evaluate(labels, X, y):
        try:
            db_score = davies_bouldin_score(X, labels)
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            print("INTERNAL EVALUATION METRICS")
            print(f'Davies Bouldin Score: {db_score}')
            print(f'Silhouette Score: {silhouette}')
            print(f'Calinski Harabasz Score: {calinski}')
            rand = rand_score(y, labels)
            fowlkes_mallows = fowlkes_mallows_score(y, labels)
            v_measure = v_measure_score(y, labels)
            print("EXTERNAL EVALUATION METRICS")
            print(f"Rand Score: {rand}")
            print(f"Fowlkes Mallows Score: {fowlkes_mallows}")
            print(f"V Measure Score: {v_measure}")
        except ValueError as e:
            print(f'Error: {e}')
