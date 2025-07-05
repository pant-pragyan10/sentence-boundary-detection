"""
utils.py
Helper utilities for entropy-based traversal and plotting.
"""
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(vec1: Any, vec2: Any) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

def plot_entropy_curve(nodes: list, entropy_values: list, save_path: str = None):
    plt.figure(figsize=(8, 4))
    plt.plot(entropy_values, marker='o')
    plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
    plt.xlabel('Node')
    plt.ylabel('Entropy')
    plt.title('Entropy Curve Across Traversal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
