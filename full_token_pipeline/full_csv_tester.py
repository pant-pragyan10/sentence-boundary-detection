"""
full_csv_tester.py
Builds a token-level Knowledge Graph (each word = node, NEXT edges between adjacent tokens) and
outputs entropy for every word in the paragraph.

This lives in `entropy/full_token_pipeline/` so it does not interfere with the main SVO-based pipeline.
"""
import os
import csv
import argparse
from typing import List, Tuple
import spacy
import networkx as nx
import numpy as np
from node2vec_embedder import Node2VecEmbedder
from utils import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def build_token_graph(text: str) -> Tuple[nx.DiGraph, List[List[str]]]:
    """Return DiGraph where each token occurrence is a unique node.

    Node label = (global_index, token_text). We also keep per-sentence token lists
    for later CSV output. Adds a synthetic START node connected to each sentence start and END node to each sentence end.
    """
    doc = nlp(text)
    graph = nx.DiGraph()
    global_idx = 0
    sentences_tokens: List[List[str]] = []
    START_NODE = "START"
    END_NODE = "SENT_END"
    graph.add_node(START_NODE, token="START")
    graph.add_node(END_NODE, token="SENT_END")

    for sent in doc.sents:
        sent_tokens: List[str] = []
        prev_node = None
        for tok in sent:
            if tok.is_punct:
                continue  # skip punctuation tokens
            node_id = f"{global_idx}:{tok.text}"
            graph.add_node(node_id, token=tok.text)
            sent_tokens.append(node_id)
            if prev_node is not None:
                graph.add_edge(prev_node, node_id, label="NEXT")
                graph.add_edge(node_id, prev_node, label="PREV")  # bidirectional for traversal
            prev_node = node_id
            global_idx += 1
        if sent_tokens:
            graph.add_edge(START_NODE, sent_tokens[0], label="SENT_START")
            graph.add_edge(END_NODE, sent_tokens[-1], label="SENT_END")
        sentences_tokens.append(sent_tokens)
    return graph, sentences_tokens


def compute_entropies(graph: nx.DiGraph, sentences_tokens: List[List[str]], embedder: Node2VecEmbedder):
    """Compute entropy for each token using max distance from START and SENT_END node embeddings."""
    embeddings = embedder.get_embeddings()
    rows = []
    sent_no = 1
    START_NODE = "START"
    END_NODE = "SENT_END"
    start_vec = embeddings.get(START_NODE)
    end_vec = embeddings.get(END_NODE)
    for sent_tokens in sentences_tokens:
        if not sent_tokens or start_vec is None or end_vec is None:
            continue
        entropies = []
        for node in sent_tokens:
            vec = embeddings.get(node)
            if vec is None:
                entropy = None
            else:
                entropy_start = 1.0 - cosine_similarity(start_vec, vec)
                entropy_end = 1.0 - cosine_similarity(end_vec, vec)
                entropy = max(entropy_start, entropy_end)
            entropies.append(entropy)
        for idx, node in enumerate(sent_tokens):
            token_text = graph.nodes[node]["token"]
            prev_word = graph.nodes[sent_tokens[idx-1]]["token"] if idx > 0 else ""
            entropy_val = entropies[idx]
            if entropy_val is not None:
                entropy_val = round(entropy_val, 3)
            rows.append({
                "Word": token_text,
                "Entropy": entropy_val if entropy_val is not None else "N/A",
                "Sentence #": sent_no,
                "Is First Word": "TRUE" if idx == 0 else "FALSE",
                "Previous Word": prev_word,
                "Sentence (truncated)": " ".join(graph.nodes[t]["token"] for t in sent_tokens)[:40]
            })
        sent_no += 1
    return rows


def main():
    """Main entry: parse args, run pipeline, write CSV."""
    parser = argparse.ArgumentParser(description="Token-level entropy-based sentence boundary detection.")
    parser.add_argument('--input', type=str, default=None, help='Input text file (default: input.txt in project root)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: output_full_tokens.csv in project root)')
    args = parser.parse_args()

    # Default paths if not provided
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input if args.input else os.path.join(script_dir, "input.txt")
    output_path = args.output if args.output else os.path.join(script_dir, "output_full_tokens.csv")

    with open(input_path, "r") as f:
        text = f.read().strip()
    graph, sentences_tokens = build_token_graph(text)
    embedder = Node2VecEmbedder(dimensions=64, walk_length=30, num_walks=120, q=16.0)
    embedder.fit(graph)
    rows = compute_entropies(graph, sentences_tokens, embedder)

    with open(output_path, "w", newline="") as csvfile:
        fieldnames = ["Word", "Entropy", "Sentence #", "Is First Word", "Previous Word", "Sentence (truncated)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Full-token CSV written to {out_path}")

if __name__ == "__main__":
    main()
