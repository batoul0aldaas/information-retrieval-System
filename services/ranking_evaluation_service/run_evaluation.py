import json
import pandas as pd
from collections import defaultdict

from services.api_gateway.index_registry import (
    get_index,
    get_embeddings
)

from services.retrieval_service.tfidf_retrieval import retrieve_tfidf
from services.retrieval_service.bm25_retrieval import retrieve_bm25
from services.retrieval_service.embedding_retrieval import retrieve_embedding
from services.retrieval_service.hybrid_retrieval import (
    retrieve_hybrid_serial,
    retrieve_hybrid_parallel
)

from services.ranking_evaluation_service.evaluator import (
    average_precision,
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)