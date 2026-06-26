flowchart TD

A[User] --> B[Streamlit UI]

B --> C[API Gateway]

C --> D[TF-IDF Retrieval]
C --> E[BM25 Retrieval]
C --> F[Embedding Retrieval]
C --> G[Hybrid Retrieval]
C --> H[Evaluation Service]

D --> I[Index Registry]
E --> I
F --> J[Embeddings Store]
G --> I
G --> J

H --> K[Metrics Report]