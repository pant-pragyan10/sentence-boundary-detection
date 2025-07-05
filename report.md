# Entropy-Based Sentence Boundary Detection in Knowledge Graphs — Report

## Overview
This project implements a novel, entropy-based approach for sentence boundary detection using Knowledge Graphs (KGs) and node embeddings. The method is inspired by the BLT (Byte Latent Tokenizer) architecture and is designed to work without access to raw text at inference time, relying solely on graph structure and learned embeddings.

---

## Approaches & Pipeline

### 1. SVO-Based Pipeline
- **SVO Extraction:**
  - Uses spaCy to extract Subject-Verb-Object triplets from input text.
- **Knowledge Graph Construction:**
  - Nodes represent subjects and objects; edges represent verbs.
- **Node Embedding:**
  - Node2Vec is used to generate embeddings for each node.
- **Entropy-Based Traversal:**
  - Bidirectional traversal from each sentence start node, computing entropy as `1 - cosine_similarity(start_node, current_node)`.
  - Traversal halts when entropy exceeds a threshold, indicating a likely sentence boundary.
- **CSV Output:**
  - Outputs a CSV with traversed nodes, entropy values, sentence numbers, and metadata.

### 2. Full-Token Pipeline (Advanced)
- **Token-Level Graph:**
  - Every non-punctuation token is a node; edges connect adjacent tokens.
  - Synthetic `START` and `SENT_END` nodes connect to the first and last word of each sentence, respectively.
- **Node2Vec Embedding:**
  - Node2Vec with high `q` parameter (e.g., 16.0) for sharper boundary detection.
- **Hybrid Entropy Reference:**
  - For each word, entropy is computed as the maximum distance from both `START` and `SENT_END` node embeddings:  
    `entropy = max(1 - cosine(START, word), 1 - cosine(SENT_END, word))`
- **Output:**
  - Produces `output_full_tokens.csv` with entropy for every word, highlighting sentence boundaries.

---

## Key Technologies Used
- **spaCy**: Linguistic parsing and SVO extraction
- **networkx**: Graph construction and manipulation
- **node2vec**: Node embedding generation
- **numpy, scipy**: Numerical operations and similarity computations
- **Python 3.x**: All code is written in Python

---

## Results & Observations
- **Entropy Spikes:**
  - The entropy signal reliably spikes at the first word of each sentence, especially in the full-token pipeline with hybrid reference and high `q`.
  - This makes sentence boundaries visually and numerically distinguishable in the output CSV.
- **Flexibility:**
  - The pipeline supports both SVO-based and token-based graph construction for comparative analysis.
- **Reproducibility:**
  - All scripts are parameterized and professional, with no hardcoded paths or debug code.
  - Only the final output CSV is retained for clarity and ease of use.

---

## Usage Summary
1. Install dependencies: `pip install -r requirements.txt`
2. Place your input text in `input.txt` (project root).
3. Run the pipeline:
   - SVO pipeline: `python csv_tester.py --input input.txt --output output_full_tokens.csv`
   - Full-token pipeline: `python full_token_pipeline/full_csv_tester.py --input input.txt --output output_full_tokens.csv`
4. Inspect `output_full_tokens.csv` for entropy values and boundary detection results.

---

## Conclusion
This project demonstrates that entropy-based traversal of token-level knowledge graphs, especially with hybrid reference anchors and tuned Node2Vec, can provide an interpretable signal for sentence boundary detection. The approach is modular, extensible, and suitable for further research or integration into larger NLP pipelines.
