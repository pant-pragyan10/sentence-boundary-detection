"""
node2vec_embedder.py
Generates node embeddings for the KG using Node2Vec.
"""
from typing import Dict
import networkx as nx
import numpy as np
from node2vec import Node2Vec

class Node2VecEmbedder:
    """Wrapper around the `node2vec` implementation with convenient hyper-parameters.

    The default hyper-parameters are tuned for better semantic capture in
    small/medium KGs representing sentences.  You can override any of them
    through the constructor.
    """

    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 20,
        num_walks: int = 100,
        p: float = 0.5,
        q: float = 2.0,
        workers: int = 1,
        directed: bool = True,
    ):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # Return parameter (controls BFS/DFS)
        self.q = q  # In-out parameter (controls exploration)
        self.workers = workers
        self.directed = directed
        self.model = None

    def fit(self, graph: nx.DiGraph):
        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,

        )
        # window=5 usually works well; sg=1 uses skip-gram
        self.model = node2vec.fit(window=5, min_count=1, sg=1)

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings = {}
        for node in self.model.wv.index_to_key:
            embeddings[node] = self.model.wv[node]
        return embeddings
