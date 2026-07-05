import numpy as np

from Strategy.SimilarityStrategy import SimilarityStrategy


class CosineStrategy(SimilarityStrategy):
    def compare(self, v1, v2):
        return np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2)
        )
