## Live Demo
🚀 https://your-app-url.streamlit.app


# E-Commerce Product Intelligence Pipeline with GenAI / RAG

An end-to-end data + GenAI pipeline built with free and open-source tools.

## Architecture
Raw CSV → PySpark → Parquet → sentence-transformers → ChromaDB → RAG → FastAPI → Streamlit

## Tech Stack
| Component | Tool | GCP Equivalent |
|---|---|---|
| Data Processing | PySpark (Docker) | Cloud Dataproc |
| Embeddings | sentence-transformers | Vertex AI Embeddings |
| Vector DB | ChromaDB | Vertex AI Vector Search |
| LLM | Groq LLaMA 3.3 | Vertex AI Gemini |
| Orchestration | Apache Airflow (Docker) | Cloud Composer |
| SQL Analytics | DuckDB | BigQuery |
| API | FastAPI | Cloud Run |
| UI | Streamlit | Cloud Run |

## Dataset
Amazon Echo Dot 2 Reviews — 6,855 reviews from Kaggle

## How to Run

### 1. Start Docker
docker compose up -d

### 2. Activate venv
source venv/bin/activate

### 3. Run PySpark jobs
spark-run /opt/spark_jobs/process_reviews.py
spark-run /opt/spark_jobs/process_products.py

### 4. Generate embeddings
python embeddings/generate_embeddings.py

### 5. Start API
uvicorn api.main:app --reload --port 8000

### 6. Start UI
streamlit run streamlit_app.py

## Author
Chinmay Kumar Rout — GCP Data Engineer
