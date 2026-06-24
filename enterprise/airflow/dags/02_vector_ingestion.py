"""
DAG 02 — Vector DB Ingestion
Silver layer documents → Qdrant (RAG knowledge base).

Reads processed documents from MinIO and pushes them to the RAG API
for embedding and storage in Qdrant.
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
}


def ingest_documents(ds: str, **_: object) -> str:
    """Pull curated documents from MinIO and send to RAG API for embedding."""
    import httpx

    rag_api_url = "http://rag-api:8000"

    # In production, read actual documents from MinIO s3a://enterprise-data/silver/<date>/
    # Here we simulate representative document chunks
    sample_documents = [
        {
            "text": f"[{ds}] Q1 sales increased 15 % YoY, driven by new product launches in EMEA.",
            "metadata": {"source": "sales_report", "date": ds, "department": "sales"},
        },
        {
            "text": f"[{ds}] Customer NPS reached 62 (industry avg 45). Top driver: response time.",
            "metadata": {"source": "customer_report", "date": ds, "department": "cx"},
        },
        {
            "text": f"[{ds}] Cloud infrastructure cost reduced 12 % through reserved-instance migration.",
            "metadata": {"source": "infra_report", "date": ds, "department": "it"},
        },
        {
            "text": f"[{ds}] HR headcount: 1 240 employees. Attrition rate: 8 % (target <10 %).",
            "metadata": {"source": "hr_report", "date": ds, "department": "hr"},
        },
    ]

    ingested = 0
    with httpx.Client(timeout=30, base_url=rag_api_url) as client:
        for doc in sample_documents:
            resp = client.post("/ingest", json=doc)
            resp.raise_for_status()
            ingested += 1

    result = f"Ingested {ingested} documents for date={ds}"
    print(result)
    return result


def validate_vector_store(**_: object) -> None:
    """Sanity-check: run a test query against the RAG API."""
    import httpx

    resp = httpx.post(
        "http://rag-api:8000/query",
        json={"query": "What are the latest sales results?", "top_k": 2},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    assert data.get("answer"), "RAG API returned empty answer"
    print(f"Validation passed. Answer preview: {data['answer'][:120]}...")


with DAG(
    dag_id="02_vector_ingestion",
    description="Ingest curated docs into Qdrant via RAG API",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["vectordb", "rag", "qdrant"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_documents",
        python_callable=ingest_documents,
    )

    validate = PythonOperator(
        task_id="validate_vector_store",
        python_callable=validate_vector_store,
    )

    ingest >> validate
