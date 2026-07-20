import time
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings
from typing import List, Dict, Any, Optional

logger = logging.getLogger("rag_system.indexing")

class EmbeddingModelSingleton:
    _instance: Optional[SentenceTransformer] = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._instance is None:
            model_name = settings.EMBEDDING_MODEL
            logger.info(f"Initializing Embedding model: {model_name} (Cache: {settings.HF_HOME})")
            start = time.time()
            try:
                cls._instance = SentenceTransformer(model_name)
                logger.info(f"Embedding model loaded successfully in {time.time() - start:.2f}s")
            except Exception as e:
                logger.error(f"Error loading embedding model {model_name}: {e}. Falling back to 'all-MiniLM-L6-v2'")
                try:
                    cls._instance = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("Fallback 'all-MiniLM-L6-v2' model loaded successfully.")
                except Exception as fallback_err:
                    logger.critical(f"Failed to load fallback embedding model: {fallback_err}")
                    raise
        return cls._instance

class ChromaManager:
    def __init__(self):
        logger.info(f"Connecting to ChromaDB at: {settings.CHROMA_DB_DIR}")
        try:
            self.client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            # Pre-load embedding model
        except Exception as e:
            logger.critical(f"ChromaDB initialization failed: {e}")
            raise

    def add_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]):
        """
        Generates embeddings for chunks and adds them to ChromaDB.
        """
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"].copy()
            meta["doc_id"] = doc_id
            meta["chunk_index"] = i
            metadatas.append(meta)
            
        logger.info(f"Generating embeddings for {len(chunks)} chunks using BGE...")
        start_time = time.time()
        try:
            model = EmbeddingModelSingleton.get_model()
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            latency = time.time() - start_time
            logger.info(f"Embeddings generation latency: {latency:.4f}s")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document {doc_id}: {e}")
            raise
        
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            logger.info(f"Successfully indexed {len(chunks)} chunks to ChromaDB for doc ID: {doc_id}")
        except Exception as e:
            logger.error(f"Error adding vector records to ChromaDB: {e}")
            raise

    def delete_document_chunks(self, doc_id: str):
        """
        Removes all chunks associated with a document.
        """
        try:
            self.collection.delete(where={"doc_id": doc_id})
            logger.info(f"Successfully purged vector segments from ChromaDB for doc ID: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting chunks for doc ID {doc_id} from ChromaDB: {e}")
            raise

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Retrieves all documents and their metadata.
        """
        try:
            results = self.collection.get(include=["metadatas", "documents"])
            chunks = []
            if results and "documents" in results and results["documents"]:
                for i in range(len(results["documents"])):
                    chunks.append({
                        "id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })
            return chunks
        except Exception as e:
            logger.error(f"Failed to query all chunks from ChromaDB: {e}")
            return []

    def query_semantic(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for semantic similarity.
        """
        try:
            model = EmbeddingModelSingleton.get_model()
            query_instruction = "Represent this sentence for searching relevant passages: "
            
            start_time = time.time()
            query_embedding = model.encode(query_instruction + query, show_progress_bar=False).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            query_latency = time.time() - start_time
            logger.info(f"ChromaDB semantic search latency: {query_latency:.4f}s")
            
            output = []
            if results and "documents" in results and results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance
                    
                    output.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": similarity,
                        "id": results["ids"][0][i]
                    })
            return output
        except Exception as e:
            logger.error(f"Failed to run semantic query in ChromaDB: {e}")
            return []
