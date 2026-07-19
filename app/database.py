import sqlite3
import logging
from app.config import settings

logger = logging.getLogger("rag_system.database")

def get_db_connection():
    try:
        conn = sqlite3.connect(settings.SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite database at {settings.SQLITE_DB_PATH}: {e}")
        raise

def init_db():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create queries table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create claims/citations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY,
            query_id TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            cited_passage TEXT,
            doc_name TEXT,
            page_number INTEGER,
            section_header TEXT,
            similarity_score REAL,
            nli_status TEXT,
            confidence_score REAL,
            is_supported INTEGER,
            FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
        )
        """)
        
        conn.commit()
        logger.info("Database tables initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error during SQLite table initialization: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Document operations
def insert_document(doc_id: str, filename: str, filepath: str, file_type: str, file_size: int, chunk_count: int = 0):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO documents (id, filename, filepath, file_type, file_size, chunk_count) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, filename, filepath, file_type, file_size, chunk_count)
        )
        conn.commit()
        logger.info(f"Indexed document metadata saved: {filename} ({chunk_count} chunks)")
    except sqlite3.Error as e:
        logger.error(f"Error inserting document metadata for {filename}: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_all_documents():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving all documents from SQLite: {e}")
        return []
    finally:
        if conn:
            conn.close()

def delete_document(doc_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        logger.info(f"Metadata deleted for doc ID: {doc_id}")
    except sqlite3.Error as e:
        logger.error(f"Error deleting document {doc_id} from SQLite: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Query & Claim operations
def insert_query(query_id: str, query_text: str, response: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO queries (id, query_text, response) VALUES (?, ?, ?)",
            (query_id, query_text, response)
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error logging query in SQLite: {e}")
        raise
    finally:
        if conn:
            conn.close()

def insert_claim(claim_id: str, query_id: str, claim_text: str, cited_passage: str, 
                 doc_name: str, page_number: int, section_header: str, 
                 similarity_score: float, nli_status: str, confidence_score: float, is_supported: bool):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO claims (
                id, query_id, claim_text, cited_passage, doc_name, page_number, 
                section_header, similarity_score, nli_status, confidence_score, is_supported
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            claim_id, query_id, claim_text, cited_passage, doc_name, page_number, 
            section_header, similarity_score, nli_status, confidence_score, 1 if is_supported else 0
        ))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error inserting audited claim logs: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_query_with_claims(query_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM queries WHERE id = ?", (query_id,))
        query_row = cursor.fetchone()
        if not query_row:
            return None
        
        cursor.execute("SELECT * FROM claims WHERE query_id = ?", (query_id,))
        claim_rows = cursor.fetchall()
        
        return {
            "query": dict(query_row),
            "claims": [dict(row) for row in claim_rows]
        }
    except sqlite3.Error as e:
        logger.error(f"Error querying claims audit data: {e}")
        return None
    finally:
        if conn:
            conn.close()
