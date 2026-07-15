# ============================================================
# RETRIEVAL — OpenSearch only (no Knowledge Base)
# ============================================================
import os
LOCAL_DEV = os.getenv("LOCAL_DEV", "false").lower() == "true"
import asyncio
import json
import logging
from typing import List, Optional

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Clients ──────────────────────────────────────────────────
bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

creds = boto3.Session().get_credentials()
auth = AWS4Auth(
    creds.access_key, creds.secret_key,
    "ap-southeast-1", "es",
    session_token=creds.token
)

opensearch = OpenSearch(
    hosts=[{"host": "vpc-job-evaluator-cluster-56l6nbvkel7yfszuvo7pcczraq.ap-southeast-1.es.amazonaws.com", "port": 443}],
    http_auth=('postgres', 'Welcome1234$$'),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

INDEX_NAME = "jobs-96-history-index"

# ── Embedding ─────────────────────────────────────────────────

async def get_embedding(text: str) -> Optional[List[float]]:
    model_ids = ['cohere.embed-english-v3', 'cohere.embed-multilingual-v3']
    loop = asyncio.get_running_loop()

    for model_id in model_ids:
        try:
            body = json.dumps({
                "texts": [text[:2048]],
                "input_type": "search_document"
            })
            response = await loop.run_in_executor(
                None,
                lambda: bedrock_runtime.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept='application/json',
                    contentType='application/json'
                )
            )
            response_body = json.loads(response.get('body').read())
            embeddings = response_body.get('embeddings', [])
            if embeddings:
                logger.info(f"Embedding generated via {model_id} ({len(embeddings[0])} dims)")
                return embeddings[0]
        except Exception as e:
            logger.warning(f"Embedding model {model_id} failed: {e}")

    logger.error("All embedding models failed")
    return None


# ── Vector search ─────────────────────────────────────────────

async def retrieve_similar_jobs(
    summary: dict,
    min_similarity: float = 0.65,
    max_results: int = 2
) -> list:
    if LOCAL_DEV:
        logger.info("LOCAL_DEV mode — skipping OpenSearch retrieval")
        return []
    loop = asyncio.get_running_loop()
    job_text = f"{summary.get('job_title', '')} {summary.get('summary', '')}"
    query_embedding = await get_embedding(job_text)

    if not query_embedding:
        logger.warning("No embedding — falling back to text search")
        return await _text_search_fallback(summary)

    try:
        knn_query = {
            "size": max_results,
            "query": {
                "knn": {
                    "job_description_vector": {
                        "vector": query_embedding,
                        "k": 10
                    }
                }
            },
            "_source": [
                "Job Title", "Job Description", "Know-How", "Approved grade",
                "KH Point", "Grand Total", "Profile",
                "KH1", "KH2", "KH3",
                "Problem Solving", "PS1", "PS2", "PS3", "PS Score",
                "Accountability", "ACC1", "ACC2", "ACC3", "ACC Point"
            ],
            "min_score": min_similarity
        }

        res = await loop.run_in_executor(
            None,
            lambda: opensearch.search(index=INDEX_NAME, body=knn_query)
        )

        hits = res.get("hits", {}).get("hits", [])
        filtered = [h for h in hits if h.get("_score", 0) >= min_similarity][:max_results]

        logger.info(
            f"Vector search: {len(hits)} raw hits, "
            f"{len(filtered)} above {min_similarity*100:.1f}% threshold"
        )

        if not filtered and hits:
            logger.info(f"Highest score below threshold: {hits[0].get('_score', 0):.3f}")

        return filtered

    except Exception as e:
        logger.warning(f"k-NN search failed: {e} — falling back to text search")
        return await _text_search_fallback(summary)


async def _text_search_fallback(summary: dict) -> list:
    loop = asyncio.get_running_loop()
    search_query = summary.get("summary", "")
    logger.info(f"Text search fallback for: '{search_query[:100]}'")

    try:
        res = await loop.run_in_executor(
            None,
            lambda: opensearch.search(
                index=INDEX_NAME,
                body={
                    "size": 2,
                    "query": {
                        "multi_match": {
                            "query": search_query,
                            "fields": ["Job Title^3", "Job Description^2"],
                            "minimum_should_match": "10%"
                        }
                    },
                    "_source": [
                        "Job Title", "Job Description", "Know-How", "Approved grade",
                        "KH Point", "Grand Total", "Profile",
                        "KH1", "KH2", "KH3",
                        "Problem Solving", "PS1", "PS2", "PS3", "PS Score",
                        "Accountability", "ACC1", "ACC2", "ACC3", "ACC Point"
                    ]
                }
            )
        )
        hits = res.get("hits", {}).get("hits", [])
        logger.info(f"Text search returned {len(hits)} results")
        return hits
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return []