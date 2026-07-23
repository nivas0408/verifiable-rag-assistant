import time
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Tuple
from app.config import settings
from app.core.indexing import EmbeddingModelSingleton
from app.core.generation import is_llm_available

logger = logging.getLogger("rag_system.verification")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

class CitationVerifier:
    def __init__(self):
        self._embedding_model = None
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        logger.info(f"CitationVerifier initialized (Provider: {self.provider.upper()}, Model: {self.model})")

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            return EmbeddingModelSingleton.get_model()
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, value):
        self._embedding_model = value

    def _nli_check_llm(self, claim: str, passage: str) -> str:
        """Runs an NLI logic check via the configured LLM provider comparing the claim against the passage."""
        if not is_llm_available():
            # Fallback NLI heuristic if LLM is offline
            claim_emb = self.embedding_model.encode(claim)
            passage_emb = self.embedding_model.encode(passage)
            sim = cosine_similarity(claim_emb, passage_emb)
            if sim >= 0.65:
                return "SUPPORTED"
            elif sim >= 0.4:
                return "UNSUPPORTED"
            else:
                return "CONTRADICTED"

        prompt = f"""You are a logical verification system. Your job is to determine if the provided Context supports the Claim. 

Compare the Claim with the Context and output exactly one of the following classification words:
- SUPPORTED: The claim is directly stated or logically implied by the context.
- CONTRADICTED: The context directly contradicts the claim.
- UNSUPPORTED: The context does not contain enough information to support or contradict the claim.

Context:
{passage}

Claim:
{claim}

Response (Only output one word - SUPPORTED, CONTRADICTED, or UNSUPPORTED):"""

        if self.provider == "groq":
            return self._nli_groq(prompt)
        else:
            return self._nli_ollama(prompt)

    def _nli_groq(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10.0)
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"].strip().upper()
                return self._parse_nli_word(text)
            return "UNSUPPORTED"
        except Exception as e:
            logger.error(f"Groq API NLI verification failed: {e}")
            return "UNSUPPORTED"

    def _nli_ollama(self, prompt: str) -> str:
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10.0)
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "").strip().upper()
                return self._parse_nli_word(text)
            return "UNSUPPORTED"
        except Exception as e:
            logger.error(f"Local Ollama NLI verification failed: {e}")
            return "UNSUPPORTED"

    def _parse_nli_word(self, text: str) -> str:
        if "SUPPORTED" in text:
            return "SUPPORTED"
        elif "CONTRADICTED" in text:
            return "CONTRADICTED"
        else:
            return "UNSUPPORTED"

    def verify_claim(self, claim: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifies a single claim against its cited passages.
        Returns verification details, status, and confidence score.
        """
        if not citations:
            return {
                "claim": claim,
                "status": "UNSUPPORTED",
                "confidence_score": 0.0,
                "similarity_score": 0.0,
                "citation_details": []
            }
            
        claim_emb = self.embedding_model.encode(claim)
        citation_details = []
        
        best_similarity = 0.0
        best_nli = "UNSUPPORTED"
        best_confidence = 0.0
        best_citation = None

        for citation in citations:
            passage = citation["chunk_text"]
            passage_emb = self.embedding_model.encode(passage)
            sim = cosine_similarity(claim_emb, passage_emb)
            
            nli_status = self._nli_check_llm(claim, passage)
            
            if nli_status == "SUPPORTED":
                confidence = 50.0 + (50.0 * sim)
            elif nli_status == "UNSUPPORTED":
                confidence = 40.0 * sim
            else:  # CONTRADICTED
                confidence = 0.0
                
            confidence = min(100.0, max(0.0, confidence))
            
            detail = {
                "doc_name": citation["doc_name"],
                "page": citation["page"],
                "section": citation["section"],
                "source_idx": citation["source_idx"],
                "similarity_score": sim,
                "nli_status": nli_status,
                "confidence_score": confidence
            }
            citation_details.append(detail)
            
            if best_citation is None or confidence > best_confidence:
                best_confidence = confidence
                best_similarity = sim
                best_nli = nli_status
                best_citation = detail

        # Determine overall claim status using settings threshold
        threshold_percentage = settings.VERIFICATION_THRESHOLD * 100
        if best_confidence >= threshold_percentage:
            overall_status = "SUPPORTED"
        elif best_nli == "CONTRADICTED":
            overall_status = "CONTRADICTED"
        else:
            overall_status = "UNSUPPORTED"

        return {
            "claim": claim,
            "status": overall_status,
            "confidence_score": best_confidence,
            "similarity_score": best_similarity,
            "best_citation": best_citation,
            "citation_details": citation_details
        }

    def verify_answer(self, parsed_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs verification on all claims in the generated answer.
        Tracks verification runtime latency.
        """
        logger.info(f"Auditing grounding verification on {len(parsed_claims)} claims...")
        start_time = time.time()
        results = []
        for claim_info in parsed_claims:
            verification = self.verify_claim(claim_info["claim"], claim_info["citations"])
            results.append(verification)
            
        latency = time.time() - start_time
        logger.info(f"Citation verification audit completed in {latency:.4f}s")
        return results
