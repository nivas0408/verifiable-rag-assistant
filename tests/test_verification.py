import pytest
import numpy as np
from unittest.mock import MagicMock
from app.core.verification import cosine_similarity, CitationVerifier
from app.core.generation import AnswerGenerator

def test_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(cosine_similarity(v1, v2), 1e-5) == 1.0
    
    v3 = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(cosine_similarity(v1, v3), 1e-5) == 0.0
    
    v4 = np.array([-1.0, 0.0, 0.0])
    assert pytest.approx(cosine_similarity(v1, v4), 1e-5) == -1.0

def test_citation_parsing():
    answer = "Transformers use self-attention [1]. This enables long-range dependencies [2][3]."
    retrieved_chunks = [
        {"text": "Chunk 1 content", "metadata": {"source": "doc1.txt"}},
        {"text": "Chunk 2 content", "metadata": {"source": "doc2.txt"}},
        {"text": "Chunk 3 content", "metadata": {"source": "doc3.txt"}}
    ]
    
    claims = AnswerGenerator.parse_citations(answer, retrieved_chunks)
    
    assert len(claims) == 2
    
    # Check claim 1
    assert "Transformers use self-attention" in claims[0]["claim"]
    assert len(claims[0]["citations"]) == 1
    assert claims[0]["citations"][0]["source_idx"] == 1
    assert claims[0]["citations"][0]["doc_name"] == "doc1.txt"
    
    # Check claim 2
    assert "This enables long-range dependencies" in claims[1]["claim"]
    assert len(claims[1]["citations"]) == 2
    c_indices = [c["source_idx"] for c in claims[1]["citations"]]
    assert 2 in c_indices
    assert 3 in c_indices

def test_verify_claim():
    verifier = CitationVerifier()
    
    # Mock embedding model to return static orthogonal/identical vectors
    mock_model = MagicMock()
    # Let encode return predefined vectors
    # claim vector: [1, 0]
    # passage vector: [1, 0] (perfect match)
    # another passage: [0, 1] (no match)
    def mock_encode(text):
        if "matching" in text or "Attention" in text:
            return np.array([1.0, 0.0])
        elif "different" in text:
            return np.array([0.0, 1.0])
        return np.array([1.0, 0.0])
        
    verifier.embedding_model = mock_model
    mock_model.encode.side_effect = mock_encode
    
    # Mock NLI check method to return SUPPORTED
    verifier._nli_check_llm = MagicMock(return_value="SUPPORTED")
    
    claim = "Attention mechanism is matching."
    citations = [
        {"doc_name": "doc1.txt", "page": 1, "section": "Intro", "chunk_text": "Attention mechanism is matching.", "source_idx": 1}
    ]
    
    res = verifier.verify_claim(claim, citations)
    
    assert res["status"] == "SUPPORTED"
    # Cosine similarity is 1.0 (predefined vectors)
    # Confidence for SUPPORTED = 50.0 + 50.0 * 1.0 = 100%
    assert res["confidence_score"] == 100.0
    
    # Test with CONTRADICTED NLI
    verifier._nli_check_llm = MagicMock(return_value="CONTRADICTED")
    res_contradicted = verifier.verify_claim(claim, citations)
    assert res_contradicted["status"] == "CONTRADICTED"
    assert res_contradicted["confidence_score"] == 0.0
