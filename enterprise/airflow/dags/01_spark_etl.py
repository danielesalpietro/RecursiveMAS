"""
DAG 01 — Spark ETL
Raw data lake → curated / silver layer.

Submits a PySpark job to the Spark cluster via Livy REST API.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "execution_timeout": timedelta(hours=2),
}


def submit_spark_job(ds: str, **_: object) -> None:
    """Submit an ETL PySpark job to Livy and wait for completion."""
    import time

    import requests

    livy_url = "http://livy:8998"
    job_payload = {
        "file": "s3a://enterprise-data/jobs/etl_main.py",
        "args": ["--date", ds],
        "conf": {
            "spark.executor.memory": "2g",
            "spark.executor.cores": "2",
            "spark.sql.shuffle.partitions": "50",
            "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
            "spark.hadoop.fs.s3a.access.key": "minioadmin",
            "spark.hadoop.fs.s3a.secret.key": "minioadmin123",
            "spark.hadoop.fs.s3a.path.style.access": "true",
        },
    }

    resp = requests.post(f"{livy_url}/batches", json=job_payload, timeout=30)
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    print(f"Spark batch submitted: id={batch_id}")

    # Poll until terminal state
    for _ in range(120):
        time.sleep(10)
        state_resp = requests.get(f"{livy_url}/batches/{batch_id}", timeout=10)
        state = state_resp.json().get("state", "unknown")
        print(f"Batch {batch_id} state: {state}")
        if state == "success":
            return
        if state in ("dead", "killed", "error"):
            raise RuntimeError(f"Spark job failed with state: {state}")

    raise TimeoutError(f"Spark batch {batch_id} did not finish in time")


with DAG(
    dag_id="01_spark_etl",
    description="Spark ETL: raw data → silver layer (via Livy)",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["etl", "spark", "livy"],
) as dag:

    run_etl = PythonOperator(
        task_id="submit_spark_etl",
        python_callable=submit_spark_job,
    )
