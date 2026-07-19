import pytest
from unittest.mock import MagicMock
from app.core.retrieval import HybridSearchEngine, tokenize

def test_tokenize():
    text = "The quick brown fox, jumps over the lazy dog!"
    tokens = tokenize(text)
    assert "quick" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    # Lowercase conversion check
    assert "the" in tokens
    # Punctuation removal check
    assert "dog" in tokens

def test_hybrid_search_rrf():
    # Setup mock ChromaManager
    mock_chroma = MagicMock()
    
    # Dense results (Chroma queries)
    dense_results = [
        {"id": "doc1_chunk_0", "text": "Deep learning is a subset of machine learning.", "metadata": {"source": "dl.txt"}, "score": 0.95},
        {"id": "doc1_chunk_1", "text": "Convolutional neural networks are used for image processing.", "metadata": {"source": "dl.txt"}, "score": 0.80}
    ]
    mock_chroma.query_semantic.return_value = dense_results
    
    # All chunks in database (for BM25 corpus)
    all_chunks = [
        {"id": "doc1_chunk_0", "text": "Deep learning is a subset of machine learning.", "metadata": {"source": "dl.txt"}},
        {"id": "doc1_chunk_1", "text": "Convolutional neural networks are used for image processing.", "metadata": {"source": "dl.txt"}},
        {"id": "doc2_chunk_0", "text": "Recurrent neural networks are used for sequential text modeling.", "metadata": {"source": "rnn.txt"}},
        {"id": "doc3_chunk_0", "text": "Keyword search BM25 is a term frequency-based algorithm.", "metadata": {"source": "bm25.txt"}}
    ]
    mock_chroma.get_all_chunks.return_value = all_chunks
    
    search_engine = HybridSearchEngine(mock_chroma)
    
    # Query matching BM25 keyword but not dense results
    query = "BM25 keyword search"
    
    # Run retrieval (without reranker)
    results = search_engine.retrieve(query, top_k_dense=2)
    
    # Verify RRF fused results
    assert len(results) > 0
    # The chunk containing 'BM25' should be returned in hybrid search since BM25 matches it perfectly
    chunk_ids = [res["id"] for res in results]
    assert "doc3_chunk_0" in chunk_ids
