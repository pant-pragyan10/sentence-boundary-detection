"""
csv_tester.py
Runs the entropy-based traversal pipeline and outputs a CSV with detailed node/word-level information per traversal.
"""
import csv
from svo_extractor import SVOExtractor
from kg_builder import KnowledgeGraphBuilder
from node2vec_embedder import Node2VecEmbedder
from entropy_traversal import EntropyTraversal

import argparse

ENTROPY_THRESHOLD = 0.4
MAX_DEPTH = 10


def main():
    parser = argparse.ArgumentParser(description="Run entropy-based SVO pipeline and output CSV.")
    parser.add_argument('--input', type=str, default='input.txt', help='Input text file')
    parser.add_argument('--output', type=str, default='output_full_tokens.csv', help='Output CSV file')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        paragraph = f.read()

    # Step 1: SVO Extraction
    svo_extractor = SVOExtractor()
    sentences = svo_extractor.split_sentences(paragraph)
    svo_triplets = svo_extractor.extract_svo_from_paragraph(paragraph)

    # Step 2: Build Knowledge Graph
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.add_svo_triplets(svo_triplets)
    graph = kg_builder.get_graph()

    # Step 3: Node2Vec Embeddings
    embedder = Node2VecEmbedder()
    embedder.fit(graph)
    embeddings = embedder.get_embeddings()

    # Step 4: Entropy-based Traversal and CSV output
    traversal = EntropyTraversal(graph, embeddings)
    csv_rows = []
    sentence_counter = 1
    for i, sentence in enumerate(sentences):
        if not svo_triplets[i]:
            continue
        start_node = svo_triplets[i][0][0]  # Subject of first SVO in sentence
        traversed_nodes, entropy_values, stop_node = traversal.traverse(
            start_node,
            entropy_threshold=ENTROPY_THRESHOLD,
            max_depth=MAX_DEPTH
        )
        prev_node = None
        for idx, (node, entropy) in enumerate(zip(traversed_nodes, entropy_values)):
            entropy_val = round(entropy, 3)
            csv_rows.append({
                'Word': node,
                'Entropy': entropy_val,
                'Sentence #': sentence_counter,
                'Is First Word': 'TRUE' if idx == 0 else 'FALSE',
                'Previous Word': prev_node if prev_node else '',
                'Sentence (truncated)': sentence[:40]
            })
            prev_node = node
        sentence_counter += 1

    # Write CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['Word', 'Entropy', 'Sentence #', 'Is First Word', 'Previous Word', 'Sentence (truncated)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
