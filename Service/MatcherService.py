import numpy as np
class MatcherService:

    def __init__(self):
        self.db = {
            "user_1": np.random.rand(512),
            "user_2": np.random.rand(512)
        }

    def cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def find_best_match(self, embedding):

        best_user = None
        best_score = 0

        for user_id, stored_emb in self.db.items():

            score = self.cosine(embedding, stored_emb)

            if score > best_score:
                best_score = score
                best_user = user_id

        return {
            "matched": best_score > 0.6,
            "user_id": best_user,
            "similarity": float(best_score)
        }
