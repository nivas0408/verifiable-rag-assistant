import pytest
from pathlib import Path
from app.core.ingestion import clean_text, detect_section_header, RecursiveMetadataChunker, DocumentIngestor

def test_clean_text():
    raw_text = "  Hello \n\n world!   This   is   RAG.  "
    assert clean_text(raw_text) == "Hello world! This is RAG."
    assert clean_text("") == ""

def test_detect_section_header():
    # Numbered heading
    assert detect_section_header("1.2 Methods of Analysis\nHere is text.", "Old Header") == "1.2 Methods of Analysis"
    # All caps heading
    assert detect_section_header("INTRODUCTION\nIn this paper...", "Old Header") == "INTRODUCTION"
    # Ordinary text should return the fallback (Old Header)
    assert detect_section_header("This is just an ordinary sentence that shouldn't match.", "Old Header") == "Old Header"

def test_chunker(tmp_path):
    # Setup mock data
    doc = {
        "text": "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.",
        "metadata": {"source": "test.txt", "page": 1, "section": "Intro"}
    }
    
    # We choose small chunk size to force split
    chunker = RecursiveMetadataChunker(chunk_size=30, chunk_overlap=10)
    chunks = chunker.chunk_documents([doc])
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 30
        assert chunk["metadata"]["source"] == "test.txt"
        assert chunk["metadata"]["page"] == 1
        assert chunk["metadata"]["section"] == "Intro"

def test_txt_ingestion(tmp_path):
    test_file = tmp_path / "test_doc.txt"
    content = """# Header 1
This is the text for section 1. It contains some statements.

# Header 2
This is the text for section 2. It contains some other statements.
"""
    test_file.write_text(content, encoding='utf-8')
    
    docs = DocumentIngestor.load_document(test_file)
    assert len(docs) >= 2
    assert docs[0]["metadata"]["section"] == "Header 1"
    assert docs[1]["metadata"]["section"] == "Header 2"
    assert "section 1" in docs[0]["text"]
    assert "section 2" in docs[1]["text"]
