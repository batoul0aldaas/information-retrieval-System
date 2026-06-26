from services.retrieval_service.embedding_retrieval import retrieve_embedding_faiss


def main() -> None:
    results = retrieve_embedding_faiss(
        query="climate change effects",
        dataset_id="dataset1",
        top_k=5,
    )

    for rank, (doc_id, score) in enumerate(results, start=1):
        print("-" * 80)
        print({
            "doc_id": doc_id,
            "score": score,
            "rank": rank,
            "model": "embedding",
        })


if __name__ == "__main__":
    main()
