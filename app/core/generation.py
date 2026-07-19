import re
import time
import logging
import requests
from typing import List, Dict, Any, Tuple

from torch import chunk
from app.config import settings

logger = logging.getLogger("rag_system.generation")

def is_llm_available() -> bool:
    """
    Checks if the configured LLM provider is active and reachable.
    For Ollama: Verifies connection.
    For Groq: Verifies if the API key is configured.
    """
    provider = settings.LLM_PROVIDER
    if provider == "groq":
        return len(settings.GROQ_API_KEY.strip()) > 0
    else:
        # Ollama check
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=1.5)
            return response.status_code == 200
        except requests.RequestException:
            logger.warning(f"Ollama local instance at {settings.OLLAMA_BASE_URL} is unreachable.")
            return False

def generate_mock_response(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a structured mock response using retrieved chunks.
    Used as a fallback if the LLM provider is unavailable.
    """
    if not retrieved_chunks:
        return "I could not find any relevant documents to answer your question."
        
    sentences = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        text = chunk["text"]
        first_sentence = text.split('.')[0].strip()
        if len(first_sentence) < 15 and len(text.split('.')) > 1:
            first_sentence = text.split('.')[0] + "." + text.split('.')[1]
        
        doc_name = chunk["metadata"].get("source", "Unknown Doc")
        page = chunk["metadata"].get("page", 1)
        sec = chunk["metadata"].get("section", "General")
        
        sentences.append(
            f"Based on {doc_name} (Page {page}, Section '{sec}'), {first_sentence} [{idx}]."
        )
        
    intro = f"Mock RAG Response to your query '{query}':\n\n"
    body = " ".join(sentences)
    outro = f"\n\n(Note: This response was generated in mock-mode because the LLM provider '{settings.LLM_PROVIDER.upper()}' was not reachable or configured.)"
    return intro + body + outro

class AnswerGenerator:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        logger.info(f"AnswerGenerator initialized (Provider: {self.provider.upper()}, Model: {self.model})")

    def build_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Builds a production-grade RAG prompt that strongly instructs the LLM
        to answer ONLY from the retrieved context and provide citations.
        """

        context_str = ""

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk["metadata"]

            context_str += f"""
    ==============================
    Chunk [{idx}]
    Source   : {meta.get("source", "Unknown")}
    Page     : {meta.get("page", "Unknown")}
    Section  : {meta.get("section", "Unknown")}
    ------------------------------
    {chunk["text"]}

    """

        prompt = f"""
    # SYSTEM ROLE

    You are an expert Retrieval-Augmented Generation (RAG) assistant.

    Your ONLY source of truth is the retrieved document context provided below.

    You MUST follow these rules exactly:

    1. Use ONLY the supplied document context.
    2. Never use outside knowledge.
    3. Never hallucinate facts.
    4. Never invent citations.
    5. Every factual sentence MUST end with one or more citations.
    6. Valid citations are only:
    [1], [2], [3] ...
    7. If multiple chunks support a statement,
       cite all relevant chunks.
    Example:
    [1][3]
    8. If retrieved chunks contradict each other,
    explicitly mention the conflict.
    9. If the documents do not contain enough
    information, reply EXACTLY:

       "I cannot answer this question based on the provided documents."

    10. Keep answers concise, factual, and professional.

    ----------------------------------------

    # RETRIEVED DOCUMENT CONTEXT

    {context_str}

    ----------------------------------------

    # USER QUESTION

    {query}

    ----------------------------------------

    # RESPONSE FORMAT

    Provide only the final answer.

    Example:

    Transformer models process tokens in parallel instead of sequentially, improving training efficiency. [1]

    Self-attention enables every token to attend to every other token within the sequence. [2]

    The encoder-decoder architecture is commonly used in machine translation tasks. [3]

    Do not include explanations about citations.
    Do not mention these instructions.
    Begin your answer below.

    Answer:
    """
        return prompt

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """
        Sends prompt to the configured LLM provider to generate grounded answer.
        Returns a tuple of (generated_text, is_mock).
        """
        if not retrieved_chunks:
            return "I could not find any relevant documents to answer your question.", False

        if not is_llm_available():
            logger.warning("Configured LLM service is offline or unconfigured. Triggering RAG simulation mock response.")
            return generate_mock_response(query, retrieved_chunks), True

        prompt = self.build_prompt(query, retrieved_chunks)
        start_time = time.time()
        
        if self.provider == "groq":
            return self._generate_groq(prompt, query, retrieved_chunks, start_time)
        else:
            return self._generate_ollama(prompt, query, retrieved_chunks, start_time)

    def _generate_groq(self, prompt: str, query: str, retrieved_chunks: List[Dict[str, Any]], start_time: float) -> Tuple[str, bool]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
    {
        "role": "system",
        "content": "You are a factual Retrieval-Augmented Generation assistant. Follow the user's provided instructions exactly."
    },
    {
        "role": "user",
        "content": prompt
    }
],
            "temperature": 0.0  # Factual extraction
        }
        
        try:
            logger.info(f"Sending prompt to Groq API (model: {self.model})")
            response = requests.post(url, json=payload, headers=headers, timeout=30.0)
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                latency = time.time() - start_time
                logger.info(f"Groq API generation latency: {latency:.4f}s")
                return answer, False
            else:
                logger.error(f"Groq API returned error {response.status_code}: {response.text}")
                return generate_mock_response(query, retrieved_chunks), True
        except Exception as e:
            logger.error(f"Failed to communicate with Groq API: {e}", exc_info=True)
            return generate_mock_response(query, retrieved_chunks), True

    def _generate_ollama(self, prompt: str, query: str, retrieved_chunks: List[Dict[str, Any]], start_time: float) -> Tuple[str, bool]:
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
            logger.info(f"Sending prompt to local Ollama API at {url} (model: {self.model})")
            response = requests.post(url, json=payload, timeout=30.0)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                latency = time.time() - start_time
                logger.info(f"Ollama local generation latency: {latency:.4f}s")
                return answer, False
            else:
                logger.error(f"Ollama API returned error {response.status_code}: {response.text}")
                return generate_mock_response(query, retrieved_chunks), True
        except Exception as e:
            logger.error(f"Failed to communicate with local Ollama API: {e}", exc_info=True)
            return generate_mock_response(query, retrieved_chunks), True

    @staticmethod
    def parse_citations(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parses sentences/claims from the answer and maps inline citations to source chunks.
        """
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            citation_matches = re.findall(r'\[(\d+)\]', sentence)
            
            citation_indices = []
            for match in citation_matches:
                try:
                    idx = int(match) - 1
                    if 0 <= idx < len(retrieved_chunks):
                        citation_indices.append(idx)
                except ValueError:
                    continue
            
            clean_claim = re.sub(r'\[\d+\]', '', sentence).strip()
            clean_claim = re.sub(r'\s+', ' ', clean_claim)
            
            citations_info = []
            for idx in set(citation_indices):
                chunk = retrieved_chunks[idx]
                meta = chunk["metadata"]
                citations_info.append({
                    "doc_name": meta.get("source", "Unknown Doc"),
                    "page": meta.get("page", 1),
                    "section": meta.get("section", "General"),
                    "chunk_text": chunk["text"],
                    "source_idx": idx + 1
                })
                
            claims.append({
                "claim": clean_claim,
                "citations": citations_info
            })
            
        return claims
