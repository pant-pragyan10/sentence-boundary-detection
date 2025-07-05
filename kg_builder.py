"""
kg_builder.py
Builds a directed Knowledge Graph (KG) from SVO triplets using networkx.
"""
from typing import List, Tuple
import networkx as nx

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_svo_triplets(self, svo_triplets: List[List[Tuple[str, str, str]]]):
        for sentence_triplets in svo_triplets:
            for subj, verb, obj in sentence_triplets:
                self.graph.add_node(subj)
                self.graph.add_node(obj)
                self.graph.add_edge(subj, obj, label=verb)

    def get_graph(self) -> nx.DiGraph:
        return self.graph
