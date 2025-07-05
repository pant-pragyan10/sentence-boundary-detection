"""
entropy_traversal.py
Performs entropy-based traversal for sentence boundary detection.
"""
from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
from utils import cosine_similarity

class EntropyTraversal:
    def __init__(self, graph: nx.DiGraph, embeddings: Dict[str, np.ndarray]):
        self.graph = graph
        self.embeddings = embeddings

    def compute_entropy(self, start_node: str, current_node: str) -> float:
        return 1.0 - cosine_similarity(self.embeddings[start_node], self.embeddings[current_node])

    def traverse(self, start_node: str, entropy_threshold: float = 0.4, max_depth: int = 10) -> Tuple[List[str], List[float], str]:
        """
        Traverses the graph bidirectionally from start_node, stopping when entropy rises above threshold.
        Returns:
            - List of traversed nodes (same sentence)
            - List of their entropy values
            - Node where traversal stopped
        """
        visited = set([start_node])
        queue = [(start_node, 0)]
        traversed_nodes = [start_node]
        entropy_values = [0.0]
        stop_node = start_node

        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                break
            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in self.embeddings:
                    entropy = self.compute_entropy(start_node, neighbor)
                    traversed_nodes.append(neighbor)
                    entropy_values.append(entropy)
                    visited.add(neighbor)
                    if entropy > entropy_threshold:
                        stop_node = neighbor
                        return traversed_nodes, entropy_values, stop_node
                    queue.append((neighbor, depth + 1))
        stop_node = traversed_nodes[-1]
        return traversed_nodes, entropy_values, stop_node
