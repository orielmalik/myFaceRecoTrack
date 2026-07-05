import numpy as np

from Strategy.SimilarityStrategy import SimilarityStrategy


class EuclideanStrategy(SimilarityStrategy):

    def compare(self, v1, v2):
        distance = np.linalg.norm(v1 - v2)
        return 1 / (1 + distance)
