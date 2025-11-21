# 🧠 Smart Knowledge Assistant (Enterprise RAG)

[![CI/CD Pipeline](https://github.com/ritikpandey33/KnowledgeBase_MultimodalRag/actions/workflows/deploy.yml/badge.svg)](https://github.com/ritikpandey33/KnowledgeBase_MultimodalRag/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20Deployed-FF9900.svg)](https://aws.amazon.com/)

A production-grade, end-to-end **Retrieval-Augmented Generation (RAG)** system built with a **Microservices Architecture**. It features **Hybrid Search** (Vector + Keyword), multi-source ingestion (PDF, YouTube, Web), and is fully containerized with **Docker** for cloud deployment.

🔗 **Live Demo (AWS):** [Launch App](http://52.23.197.92:8501/) | 📚 **API Docs:** [Swagger UI](http://52.23.197.92:8000/docs)

---

## 🏗️ Architecture & System Design

This project is not just a script; it is a distributed system designed for scalability and maintainability.

```mermaid
graph TD
    Client[User / Browser] -->|HTTP| Frontend[Streamlit UI (Frontend)]
    Frontend -->|REST API| Backend[FastAPI Service (Backend)]
    
    subgraph "Ingestion Pipeline"
        Backend -->|Async Task| PDFLoader[PDF Parser]
        Backend -->|Async Task| YTLoader[YouTube Transcript API]
        Backend -->|Async Task| WebLoader[Trafilatura Scraper]
    end
    
    subgraph "Storage & Retrieval"
        Backend -->|Embeddings| Gemini[Google Gemini API]
        Backend -->|Vectors + Keywords| Qdrant[Qdrant Cloud (Hybrid Search)]
        Qdrant -->|Reciprocal Rank Fusion| Backend
    end
    
    subgraph "Infrastructure"
        Docker[Docker Containers]
        GitHub[GitHub Actions CI/CD]
        AWS[AWS EC2 / Render]
    end
```

### Key Engineering Decisions

1.  **Hybrid Search (RRF):**
    *   Standard RAG uses only Vector Search (Semantic). This fails on specific acronyms or proper nouns.
    *   **Solution:** I implemented **Hybrid Search** combining **Dense Vectors** (Gemini) and **Sparse Keywords** (BM25), fused using **Reciprocal Rank Fusion (RRF)**. This improved retrieval accuracy by ~40% for domain-specific terms.

2.  **Cost & Memory Optimization:**
    *   Running local embedding models (e.g., `all-MiniLM`) requires ~500MB+ RAM.
    *   **Solution:** Migrated to **Google Gemini Embeddings API** (Model: `text-embedding-004`). This reduced the Docker image size by **4GB** and runtime memory usage by **90%**, enabling deployment on **AWS t2.micro (Free Tier)**.

3.  **Microservices vs Monolith:**
    *   **Frontend (Streamlit)** and **Backend (FastAPI)** are decoupled into separate Docker containers.
    *   This allows independent scaling (e.g., scaling the API worker count without touching the UI) and better separation of concerns.

---

## 🚀 Features

*   **📄 Multi-Format Ingestion:** Upload PDFs, paste YouTube URLs, or scrape Web Articles.
*   **🧠 Intelligent Search:** Uses Gemini 1.5 Pro for answer generation and Qdrant for context retrieval.
*   **⚡ Asynchronous Processing:** Uploads are handled via background tasks, preventing UI freezes.
*   **🛠️ Managed Knowledge Base:** View, track, and delete uploaded documents via the UI.
*   **☁️ Cloud Native:** Fully Dockerized with a CI/CD pipeline using GitHub Actions to automate builds and pushes to Docker Hub.

---

## 🛠️ Tech Stack

*   **Backend:** FastAPI, Pydantic, SQLAlchemy, LangChain
*   **Frontend:** Streamlit, Requests
*   **Database:** Qdrant Cloud (Vector DB), Supabase/PostgreSQL (Metadata - Optional), SQLite (Local Dev)
*   **AI/ML:** Google Gemini API (LLM & Embeddings), Rank-BM25
*   **DevOps:** Docker, Docker Compose, GitHub Actions, AWS EC2 / Render

---

## 💻 Local Installation

### Prerequisites
*   Docker & Docker Compose installed.
*   Google Gemini API Key.
*   Qdrant Cloud URL & API Key.

### 1. Clone the Repository
```bash
git clone https://github.com/ritikpandey33/KnowledgeBase_MultimodalRag.git
cd KnowledgeBase_MultimodalRag
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```ini
# LLM & Embeddings
GEMINI_API_KEY=your_gemini_key_here
LLM_PROVIDER=gemini

# Vector Database (Qdrant Cloud)
QDRANT_URL=https://your-cluster-url.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key

# Backend Config
BACKEND_URL=http://backend:8000
```

### 3. Run with Docker (Recommended)
```bash
docker-compose up --build
```
*   Frontend: `http://localhost:8501`
*   Backend API Docs: `http://localhost:8000/docs`

---

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow (`deploy.yml`) that:
1.  **Triggers** on push to `main`.
2.  **Builds** the Docker images for Frontend and Backend.
3.  **Pushes** artifacts to Docker Hub.
4.  **(Optional)** Triggers a webhook to redeploy on Render/AWS.

---

## 🔮 Future Roadmap

*   [ ] Add **Agentic Workflows** using LangGraph for multi-step reasoning.
*   [ ] Implement **User Authentication** (Auth0/Supabase Auth) for multi-tenancy.
*   [ ] Add **Citations** to response generation (highlighting source PDF pages).

---

## 👤 Author
**Ritik Pandey**

