"""
main.py
Orchestrates the pipeline for entropy-based sentence boundary detection in KGs.
"""
import argparse
from svo_extractor import SVOExtractor
from kg_builder import KnowledgeGraphBuilder
from node2vec_embedder import Node2VecEmbedder
from entropy_traversal import EntropyTraversal
from utils import plot_entropy_curve


def main():
    parser = argparse.ArgumentParser(description="Entropy-based Sentence Boundary Detection in KGs")
    parser.add_argument('--input', type=str, required=True, help='Input text file (paragraph)')
    parser.add_argument('--entropy-threshold', type=float, default=0.4, help='Entropy threshold for boundary detection')
    parser.add_argument('--max-depth', type=int, default=10, help='Max traversal depth')
    parser.add_argument('--visualize', action='store_true', help='Plot entropy curves for each traversal')
    args = parser.parse_args()

    # Read input paragraph
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

    # Step 4: Entropy-based Traversal
    traversal = EntropyTraversal(graph, embeddings)
    print("\n--- Sentence Boundary Detection Results ---\n")
    for i, sentence in enumerate(sentences):
        if not svo_triplets[i]:
            continue
        start_node = svo_triplets[i][0][0]  # Subject of first SVO in sentence
        traversed_nodes, entropy_values, stop_node = traversal.traverse(
            start_node,
            entropy_threshold=args.entropy_threshold,
            max_depth=args.max_depth
        )
        print(f"Sentence {i+1} (start node: {start_node}):")
        print(f"  Traversed nodes: {traversed_nodes}")
        print(f"  Entropy values: {[round(e, 3) for e in entropy_values]}")
        print(f"  Stopped at node: {stop_node}\n")
        if args.visualize:
            plot_entropy_curve(traversed_nodes, entropy_values)

if __name__ == "__main__":
    main()
