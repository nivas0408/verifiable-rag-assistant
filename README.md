# 🛡️ Verifiable RAG Assistant

> A production-ready Retrieval-Augmented Generation (RAG) system with **Groq-powered LLM inference**, **hybrid retrieval**, and **claim-level citation verification** for trustworthy AI-generated responses.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-blueviolet)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

**Verifiable RAG Assistant** is a production-ready Retrieval-Augmented Generation (RAG) system that enables users to upload documents, build a searchable knowledge base, and receive AI-generated answers backed by verifiable citations.

Unlike conventional RAG systems, this application performs **claim-level citation verification**, ensuring that each generated statement is grounded in the retrieved source documents. It combines semantic search, hybrid retrieval, reranking, and automated grounding audits to improve transparency and trustworthiness.

---

## 📸 Application Preview

![Application](assets/application.png)

---

# ✨ Features

- 📄 Multi-format document ingestion (PDF, DOCX, PPTX, TXT, HTML, CSV, JSON, Markdown)
- 🔍 Hybrid Retrieval (Semantic Search + BM25)
- 🎯 Cross-Encoder reranking using BAAI reranker
- 🤖 Groq-powered LLM inference (default)
- 🖥️ Optional Ollama support for offline/local execution
- 📚 Retrieval-Augmented Generation (RAG)
- 🛡️ Claim-level citation verification
- 📊 Grounding audit dashboard
- ⚡ FastAPI backend
- 🎨 Interactive Streamlit frontend
- 🐳 Docker support
- 📈 Health monitoring endpoint

---

# 🏗️ Architecture

```text
                 User
                  │
                  ▼
          Streamlit Frontend
                  │
                  ▼
            FastAPI Backend
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
 Document     Hybrid Search    LLM
 Parsing    (Vector + BM25)   (Groq)
      │           │            │
      └───────────┼────────────┘
                  ▼
         Cross-Encoder Reranker
                  │
                  ▼
        Citation Verification
                  │
                  ▼
         Verified AI Response
```

---

# 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Groq (default), Ollama (optional) |
| Vector Database | ChromaDB |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| Retrieval | Hybrid (Vector + BM25) |
| Database | SQLite |
| Document Parsing | PyMuPDF, python-docx, python-pptx, BeautifulSoup |
| Deployment | Docker, Docker Compose |

---

# 🚀 Local Installation

## Prerequisites

- Python **3.12** (recommended)
- A **Groq API Key**

Create a Groq API key from:

https://console.groq.com/

---

## 1. Clone the Repository

```bash
git clone https://github.com/nivas0408/verifiable-rag-assistant.git

cd verifiable-rag-assistant
```

---

## 2. Configure Environment Variables

Copy:

```bash
cp .env.example .env
```

Configure:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

> **Optional:** To run locally with Ollama instead of Groq:

```env
LLM_PROVIDER=ollama
```

---

## 3. Create Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Start the Backend

```bash
uvicorn app.main:app --reload
```

Backend:

```
http://localhost:8000
```

Swagger UI:

```
http://localhost:8000/docs
```

---

## 6. Start the Frontend

```bash
streamlit run frontend/app.py
```

Frontend:

```
http://localhost:8501
```

---

# 🐳 Docker

Build and run:

```bash
docker-compose up --build
```

Services:

| Service | URL |
|----------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Health API | http://localhost:8000/health |

---

# 🌐 Deployment

## Backend (Render)

Build Command

```bash
pip install -r requirements.txt
```

Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Environment Variables

```env
LLM_PROVIDER=groq
GROQ_API_KEY=YOUR_GROQ_API_KEY
LOG_LEVEL=INFO
```

---

## Frontend (Render or Streamlit Community Cloud)

Environment Variables

```env
BACKEND_URL=https://your-backend-url.onrender.com
FRONTEND_TIMEOUT=60
```

---

# 📡 API Endpoints

| Endpoint | Description |
|-----------|-------------|
| `/docs` | Swagger API Documentation |
| `/health` | System Health |
| `/upload` | Upload Documents |
| `/query` | Ask Questions |

---

# 📊 Health Metrics

Example:

```json
{
  "status": "healthy",
  "llm_available": true,
  "metrics": {
    "document_count": 2,
    "chunk_count": 86,
    "settings": {
      "llm_provider": "GROQ"
    }
  }
}
```

---

# 🔮 Future Improvements

- [ ] Streaming LLM responses
- [ ] OCR support for scanned PDFs
- [ ] Authentication & user accounts
- [ ] Cloud vector database support
- [ ] Semantic chunk visualization
- [ ] Adjustable hybrid retrieval weights
- [ ] Multi-user document management

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Mulluri Nivas**

- GitHub: https://github.com/nivas0408
- LinkedIn: https://www.linkedin.com/in/mulluri-nivas-09053a31a/


---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
