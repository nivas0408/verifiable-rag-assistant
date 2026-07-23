import re
import time
import logging
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.config import settings
from app.core.indexing import ChromaManager

logger = logging.getLogger("rag_system.retrieval")


class RerankerModelSingleton:
    _instance: Optional[CrossEncoder] = None

    @classmethod
    def get_model(cls) -> Optional[CrossEncoder]:
        if cls._instance is None:
            model_name = settings.RERANKER_MODEL
            if not model_name or not model_name.strip():
                logger.info("Reranker is disabled (model name is empty).")
                return None
            logger.info(f"Initializing CrossEncoder reranker: {model_name}")
            start = time.time()
            try:
                try:
                    import torch
                    torch.set_num_threads(1)
                except ImportError:
                    pass
                cls._instance = CrossEncoder(model_name, device="cpu")
                logger.info(f"Reranker model loaded successfully in {time.time() - start:.2f}s")
            except Exception as e:
                logger.error(f"Error loading reranker model {model_name}: {e}. Reranking disabled.")
                cls._instance = None
        return cls._instance


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r"\w+", text.lower())


class HybridSearchEngine:
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager

    def _get_bm25_retriever(self, all_chunks: List[Dict[str, Any]]) -> Optional[BM25Okapi]:
        if not all_chunks:
            return None

        tokenized_corpus = [tokenize(chunk["text"]) for chunk in all_chunks]
        return BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k_dense: int = None) -> List[Dict[str, Any]]:
        top_k_dense = top_k_dense or settings.RETRIEVAL_TOP_K

        logger.info(
            f"Initiating hybrid search for query: '{query}' (top_k={top_k_dense})"
        )

        # Dense Retrieval
        dense_results = self.chroma_manager.query_semantic(
            query,
            top_k=top_k_dense,
        )

        all_chunks = self.chroma_manager.get_all_chunks()

        if not all_chunks:
            logger.warning("Knowledge base is empty.")
            return []

        # BM25 Retrieval
        start_bm25 = time.time()

        bm25 = self._get_bm25_retriever(all_chunks)

        sparse_results = []

        if bm25:
            tokenized_query = tokenize(query)

            scores = bm25.get_scores(tokenized_query)

            for idx, score in enumerate(scores):
                if score > 0:
                    sparse_results.append(
                        {
                            "chunk": all_chunks[idx],
                            "score": float(score),
                        }
                    )

            sparse_results = sorted(
                sparse_results,
                key=lambda x: x["score"],
                reverse=True,
            )[:top_k_dense]

        logger.info(
            f"BM25 retrieval completed in {time.time()-start_bm25:.4f}s with {len(sparse_results)} matches"
        )

        # Reciprocal Rank Fusion
        start_rrf = time.time()

        dense_ranks = {
            res["id"]: idx + 1
            for idx, res in enumerate(dense_results)
        }

        sparse_ranks = {
            res["chunk"]["id"]: idx + 1
            for idx, res in enumerate(sparse_results)
        }

        id_to_chunk = {}

        for res in dense_results:
            id_to_chunk[res["id"]] = {
                "text": res["text"],
                "metadata": res["metadata"],
                "id": res["id"],
            }

        for res in sparse_results:
            cid = res["chunk"]["id"]

            if cid not in id_to_chunk:
                id_to_chunk[cid] = {
                    "text": res["chunk"]["text"],
                    "metadata": res["chunk"]["metadata"],
                    "id": cid,
                }

        rrf_scores = {}

        k = 60

        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        for cid in all_ids:
            score = 0

            if cid in dense_ranks:
                score += 1 / (k + dense_ranks[cid])

            if cid in sparse_ranks:
                score += 1 / (k + sparse_ranks[cid])

            rrf_scores[cid] = score

        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )

        fused_results = []

        for cid in sorted_ids:
            chunk = id_to_chunk[cid]
            chunk["rrf_score"] = rrf_scores[cid]
            fused_results.append(chunk)

        logger.info(
            f"RRF score fusion completed in {time.time()-start_rrf:.4f}s."
        )

        return fused_results

    def retrieve_and_rerank(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves candidate chunks via hybrid search, and reranks them using CrossEncoder if enabled.
        """
        candidates = self.retrieve(
            query,
            top_k_dense=settings.RETRIEVAL_TOP_K,
        )

        if not candidates:
            return []

        reranker = RerankerModelSingleton.get_model()
        if reranker is None:
            logger.info("Returning RRF results without CrossEncoder reranking.")
            return candidates[: settings.RERANK_TOP_K]

        pairs = [(query, chunk["text"]) for chunk in candidates]
        logger.info(f"Reranking {len(candidates)} candidates using Cross-Encoder...")
        start_rerank = time.time()
        try:
            rerank_scores = reranker.predict(pairs)
            for idx, score in enumerate(rerank_scores):
                candidates[idx]["rerank_score"] = float(score)
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            logger.info(f"Cross-Encoder reranking latency: {time.time() - start_rerank:.4f}s")
            return reranked[: settings.RERANK_TOP_K]
        except Exception as e:
            logger.error(f"Reranking error: {e}. Returning un-reranked RRF candidates.")
            return candidates[: settings.RERANK_TOP_K]