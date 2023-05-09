'''
Author: lihaitao
Date: 2023-05-09 22:21:34
LastEditors: Do not edit
LastEditTime: 2023-05-09 22:21:34
FilePath: /lht/GitHub_code/sailer_old/src/dense/faiss_retriever/retriever.py
'''
import numpy as np
import faiss

import logging
logger = logging.getLogger(__name__)


class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray):
        faiss.normalize_L2(init_reps)
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def search(self, q_reps: np.ndarray, k: int):
        faiss.normalize_L2(q_reps)
        return self.index.search(q_reps, k)

    def add(self, p_reps: np.ndarray):
        faiss.normalize_L2(p_reps)
        self.index.add(p_reps)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in range(0, num_query, batch_size):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices