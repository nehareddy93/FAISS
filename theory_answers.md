# FAISS Theory Questions

## Q1: Difference between IndexFlatL2 and IndexFlatIP

IndexFlatL2 uses Euclidean (L2) distance to measure similarity between vectors. Lower distance means more similar vectors.

IndexFlatIP uses Inner Product (dot product) to measure similarity. Higher value means more similar vectors.

When embeddings are normalized, IndexFlatIP behaves like cosine similarity. IndexFlatL2 measures straight-line distance regardless of direction.

IndexFlatL2 is used when exact geometric distance matters. IndexFlatIP is preferred for semantic search when embeddings are normalized.


## Q2: Why normalize embeddings before adding to FAISS?

Embeddings have both magnitude and direction. Cosine similarity depends only on direction, not magnitude.

Normalization converts all vectors to unit length so comparisons depend only on direction.

After normalization, inner product becomes equivalent to cosine similarity.

Without normalization, vectors with larger magnitude can appear falsely more similar.


## Q3: What is Approximate Nearest Neighbour (ANN)?

ANN is a method used to quickly find vectors that are very close to a query without checking every vector.

Instead of guaranteeing the exact nearest neighbor, it finds a very close match much faster.

This tradeoff is used because exact search becomes too slow for large datasets.

ANN is widely used in real-world systems like recommendation engines and semantic search because it balances speed and accuracy.
