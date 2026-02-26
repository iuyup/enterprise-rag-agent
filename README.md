# Enterprise RAG Agent (MVP)

Enterprise Knowledge Base QA Agent built with Retrieval-Augmented Generation (RAG).

## Features
- Document upload (PDF/TXT/Markdown)
- Text cleaning & chunking
- Pluggable Embedding Provider (API-based)
- FAISS vector store (local + persistent)
- TopK retrieval
- RAG prompt building
- Streamlit demo UI

## Quick Start

### 1) Create venv & install
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt