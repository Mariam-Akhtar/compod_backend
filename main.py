from dotenv import load_dotenv
load_dotenv()

import boto3
sts = boto3.client("sts", region_name="ap-southeast-1")
print(sts.get_caller_identity())

import base64
import json
import urllib.parse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import pipeline   # your existing pipeline function

app = FastAPI(title="KF Job Evaluation API")


class JobDescRequest(BaseModel):
    job_desc_base64: str


class JobDescRawRequest(BaseModel):
    job_desc: str


@app.post("/api/jobeval-poc-1")
@app.post("/v1/jobeval-poc-1")
async def evaluate_job(request: JobDescRequest):
    """
    Mirrors the Lambda handler — accepts base64-encoded job description.
    """
    try:
        decoded = base64.b64decode(request.job_desc_base64).decode("utf-8")
        decoded = urllib.parse.unquote(decoded)
        job_description = decoded.strip().strip('"')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 input: {e}")

    result = await pipeline(job_description)

    return {
        "status": 200,
        "assessment": base64.b64encode(
            json.dumps(result).encode()
        ).decode()
    }


@app.post("/evaluate/raw")
async def evaluate_job_raw(request: JobDescRawRequest):
    """
    Convenience endpoint for local testing — accepts plain text directly.
    """
    result = await pipeline(request.job_desc)
    return {"status": 200, "result": result}


@app.get("/health")
async def health():
    return {"status": "ok"}