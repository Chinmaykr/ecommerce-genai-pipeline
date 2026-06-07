"""
Airflow DAG — E-Commerce GenAI Pipeline
Orchestrates: validate → reviews → products → embeddings → verify
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner"           : "chinmay",
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id            = "ecommerce_genai_pipeline",
    description       = "End-to-end RAG pipeline",
    default_args      = default_args,
    start_date        = datetime(2024, 1, 1),
    schedule          = None,
    catchup           = False,
    tags              = ["ecommerce", "genai", "rag"],
) as dag:

    def validate_inputs():
        import os
        path = "/opt/airflow/data/reviews.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        print(f"reviews.csv found — {os.path.getsize(path):,} bytes")

    def verify_outputs():
        import os, chromadb
        for path in ["/opt/airflow/output/reviews_clean", "/opt/airflow/output/products_clean"]:
            files = [f for f in os.listdir(path) if f.endswith(".parquet")]
            if not files:
                raise Exception(f"No parquet files in {path}")
            print(f"OK: {path}")
        client = chromadb.PersistentClient(path="/opt/airflow/data/chromadb")
        count  = client.get_collection("reviews").count()
        if count == 0:
            raise Exception("ChromaDB empty!")
        print(f"ChromaDB OK: {count} vectors")

    def run_embeddings():
        import subprocess, sys
        result = subprocess.run(
            [sys.executable,
             "/opt/airflow/ecommerce-genai-pipeline/embeddings/generate_embeddings.py"],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            raise Exception(result.stderr)

    t1 = PythonOperator(task_id="validate_input",       python_callable=validate_inputs)
    t2 = BashOperator(  task_id="run_pyspark_reviews",  bash_command="/opt/spark/bin/spark-submit /opt/spark_jobs/process_reviews.py")
    t3 = BashOperator(  task_id="run_pyspark_products", bash_command="/opt/spark/bin/spark-submit /opt/spark_jobs/process_products.py")
    t4 = PythonOperator(task_id="run_embedding_pipeline", python_callable=run_embeddings)
    t5 = PythonOperator(task_id="verify_output",        python_callable=verify_outputs)

    t1 >> t2 >> t3 >> t4 >> t5
