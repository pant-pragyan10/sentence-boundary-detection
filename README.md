# Entropy-based Sentence Boundary Detection in Knowledge Graphs


<img width="499" alt="image" src="https://github.com/user-attachments/assets/07fbae3f-108f-4cf0-8edd-e411c77f0d3f" />

<img width="424" alt="image" src="https://github.com/user-attachments/assets/cf99cdb5-3258-4194-91e6-7db5729bf0d2" />

<img width="424" alt="image" src="https://github.com/user-attachments/assets/f104c8d4-fe69-4d3d-a0ca-0e2ec9189700" />


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
