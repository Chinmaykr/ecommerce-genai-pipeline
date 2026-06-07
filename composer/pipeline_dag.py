"""
Airflow DAG — E-Commerce GenAI Pipeline
Orchestrates all phases in order:
  validate → reviews → products → embeddings → verify
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os

# ── Default args ──────────────────────────────────────────────────────────────

default_args = {
    "owner"           : "chinmay",
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=2),
    "email_on_failure": False,
}

# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id            = "ecommerce_genai_pipeline",
    description       = "End-to-end RAG pipeline: PySpark → Embeddings → ChromaDB",
    default_args      = default_args,
    start_date        = datetime(2024, 1, 1),
    schedule_interval = None,          # manual trigger only
    catchup           = False,
    tags              = ["ecommerce", "genai", "rag"],
) as dag:

    # ── Task 1: Start ─────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Task 2: Validate input files ──────────────────────────────────────────
    def validate_inputs():
        import os
        reviews_path = "/opt/airflow/data/reviews.csv"
        if not os.path.exists(reviews_path):
            raise FileNotFoundError(f"Missing: {reviews_path}")
        size = os.path.getsize(reviews_path)
        print(f"reviews.csv found — {size:,} bytes")
        print("Input validation passed")

    validate_input = PythonOperator(
        task_id         = "validate_input",
        python_callable = validate_inputs
    )

    # ── Task 3: PySpark — process reviews ─────────────────────────────────────
    run_pyspark_reviews = BashOperator(
        task_id      = "run_pyspark_reviews",
        bash_command = "/opt/spark/bin/spark-submit /opt/spark_jobs/process_reviews.py",
    )

    # ── Task 4: PySpark — derive products ─────────────────────────────────────
    run_pyspark_products = BashOperator(
        task_id      = "run_pyspark_products",
        bash_command = "/opt/spark/bin/spark-submit /opt/spark_jobs/process_products.py",
    )

    # ── Task 5: Generate embeddings → ChromaDB ────────────────────────────────
    def run_embeddings():
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "/opt/airflow/ecommerce-genai-pipeline/embeddings/generate_embeddings.py"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            raise Exception(f"Embedding pipeline failed:\n{result.stderr}")
        print("Embeddings generated successfully")

    run_embedding_pipeline = PythonOperator(
        task_id         = "run_embedding_pipeline",
        python_callable = run_embeddings
    )

    # ── Task 6: Verify outputs ────────────────────────────────────────────────
    def verify_outputs():
        import os
        import chromadb

        # Check Parquet files exist
        reviews_parquet  = "/opt/airflow/output/reviews_clean"
        products_parquet = "/opt/airflow/output/products_clean"

        for path in [reviews_parquet, products_parquet]:
            files = [f for f in os.listdir(path) if f.endswith(".parquet")]
            if not files:
                raise Exception(f"No Parquet files found in {path}")
            print(f"Parquet OK: {path} — {len(files)} file(s)")

        # Check ChromaDB
        client     = chromadb.PersistentClient(path="/opt/airflow/data/chromadb")
        collection = client.get_collection("reviews")
        count      = collection.count()

        if count == 0:
            raise Exception("ChromaDB collection is empty!")
        print(f"ChromaDB OK: {count} vectors indexed")
        print("All outputs verified successfully")

    verify_output = PythonOperator(
        task_id         = "verify_output",
        python_callable = verify_outputs
    )

    # ── Task 7: Done ──────────────────────────────────────────────────────────
    end = EmptyOperator(task_id="end")

    # ── Dependencies ──────────────────────────────────────────────────────────
    start >> validate_input >> run_pyspark_reviews >> run_pyspark_products >> run_embedding_pipeline >> verify_output >> end
