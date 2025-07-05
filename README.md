# Entropy-based Sentence Boundary Detection in Knowledge Graphs

This project implements an entropy-based traversal model for detecting sentence boundaries within a Knowledge Graph (KG), inspired by the BLT (Byte Latent Tokenizer) architecture.

## Structure
- `svo_extractor.py`: Extracts SVO triplets from text using spaCy.
- `kg_builder.py`: Builds a directed KG from SVO triplets.
- `node2vec_embedder.py`: Generates node embeddings using Node2Vec.
- `entropy_traversal.py`: Performs entropy-based traversal for sentence boundary detection.
- `utils.py`: Helper utilities (cosine similarity, plotting, etc.).
- `main.py`: Pipeline orchestration and CLI.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python main.py --input <input_text_file>`

---

## Requirements
See `requirements.txt` for dependencies.
