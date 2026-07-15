# ============================================================
# PIPELINE — parallel dimension evaluation + judge
# ============================================================

import asyncio
import base64
import json
import logging
import urllib.parse
from datetime import datetime

import boto3

from prompts import (
    get_know_how_prompt,
    # get_know_how_poi_prompt,
    # get_know_how_cis_prompt,
    get_problem_solving_prompt,
    get_accountability_prompt,
    get_judge_prompt,
    KNOW_HOW_TOOL,
    PROBLEM_SOLVING_TOOL,
    ACCOUNTABILITY_TOOL,
    JUDGE_TOOL,
)
from retrieval import get_embedding, retrieve_similar_jobs, opensearch, INDEX_NAME
from utils import extract_json, build_history_text, build_history_rows

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Model IDs ─────────────────────────────────────────────────
HAIKU  = "anthropic.claude-3-haiku-20240307-v1:0"
SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"

bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-southeast-1")


# ── Model invocation ──────────────────────────────────────────

async def invoke_model(model_id: str, prompt: str) -> str:
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: bedrock_runtime.invoke_model(
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }),
            modelId=model_id
        )
    )
    body = json.loads(response["body"].read())
    return "".join([c.get("text", "") for c in body.get("content", [])])

# ── Model invocation ──────────────────────────────────────────

async def converse_model(model_id: str, prompt: str, tool_config: dict = None) -> str:
    loop = asyncio.get_running_loop()

    if tool_config:
        response = await loop.run_in_executor(
            None,
            lambda: bedrock_runtime.converse(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": 4096,
                    "temperature": 0.0
                },
                toolConfig=tool_config,
            )
        )
        for block in response["output"]["message"]["content"]:
            if block.get("toolUse"):
                return json.dumps(block["toolUse"]["input"])
        logger.warning("Tool use block not found in converse response")
        return "{}"

    else:
        response = await loop.run_in_executor(
            None,
            lambda: bedrock_runtime.invoke_model(
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                }),
                modelId=model_id
            )
        )
        body = json.loads(response["body"].read())
        return "".join([c.get("text", "") for c in body.get("content", [])])


# ── Step 1: Summarise ─────────────────────────────────────────

async def summarise_job(job_desc: str) -> dict:
    prompt = f"""You are an HR analyst. Read the job description and extract a concise 
summary for Korn Ferry job evaluation purposes.

Extract only what is explicitly stated. Write "Not stated" if missing.

OUTPUT — respond with ONLY this JSON:

{{
  "job_title": "",
  "department": "",
  "reports_to": "",
  "direct_reports": "",
  "geographical_scope": "",
  "role_purpose": "",
  "top_responsibilities": [
    {{"responsibility": "", "ownership": "", "impact": ""}}
  ],
  "min_qualification": "",
  "min_years_experience": "",
  "technical_domains": "",
  "supervisory_responsibility": "",
  "team_nature": "",
  "stakeholders": {{
    "internal": "",
    "external": "",
    "engagement_type": ""
  }},
  "decision_making": "",
  "primary_accountability": "",
  "impact_level": "",
  "kf_flags": {{
    "policy_setting": "",
    "global_remit": "",
    "external_representation": "",
    "mastery_level": "",
    "thought_leadership": ""
  }}
}}

Job Description:
{job_desc}"""
    text = await invoke_model(SONNET, prompt)
    return extract_json(text)


# ── Step 2: Retrieve history ──────────────────────────────────

async def retrieve_history(summary: dict) -> tuple[str, list]:
    hits = await retrieve_similar_jobs(summary)
    hist_text, hist_data = build_history_text(hits)
    return hist_text, hist_data


# ── Step 3: Parallel dimension evaluation ────────────────────

async def evaluate_dimensions_parallel(
    job_desc: str , job_desc_full: str
) -> tuple[dict, dict, dict]:
    """
    Fire all three dimension evaluations concurrently.
    Each uses Sonnet with its own focused prompt.
    """
    history_context = ''
    # print(f"what is job description full {job_desc}")
    kh_prompt  = get_know_how_prompt(job_desc, history_context)
    # kh_ptk_prompt = get_know_how_ptk_prompt(job_desc, history_context)
    # kh_po_prompt = get_know_how_poi_prompt(job_desc, history_context)
    # kh_cis_prompt = get_know_how_cis_prompt(job_desc, history_context)
    ps_prompt  = get_problem_solving_prompt(job_desc, history_context)
    acc_prompt = get_accountability_prompt(job_desc, history_context)

    kh_text, ps_text, acc_text , (hist_text_str, hist_list)= await asyncio.gather(
        # invoke_model(HAIKU, kh_ptk_prompt),
        # invoke_model(HAIKU, kh_po_prompt),
        invoke_model(SONNET, kh_prompt),
        invoke_model(SONNET, ps_prompt),
        invoke_model(SONNET, acc_prompt),
        retrieve_history(job_desc)
    )

    # kh_ptk_result  = extract_json(kh_ptk_text)
    # kh_poi_result = extract_json(kh_mgmt_text)
    # kh_cis_result   = extract_json(kh_hr_text)
    #  # Merge the three KH sub-factors into one dict
    # kh_result = {
    #     "know_how_details": {
    #         "PTK":  kh_ptk_result.get("PTK", {}),
    #         "POI": kh_mgmt_result.get("POI", {}),
    #         "CIS":  kh_hr_result.get("CIS", {}),
    #     }
    # }
    kh_result  = extract_json(kh_text)
    ps_result  = extract_json(ps_text)
    acc_result = extract_json(acc_text)
    # hist_text =  hist_data

    logger.info(f"kh_prompt : {kh_prompt}")
    logger.info(f"kh_result : {kh_result}")
    logger.info(f"ps_prompt : {ps_prompt}")
    logger.info(f"ps_result : {ps_result}")
    logger.info(f"acc_prompt : {acc_prompt}")
    logger.info(f"acc_result : {acc_result}")
    logger.info("Parallel dimension evaluation complete")
    logger.info(f"hist_data {hist_text_str}")
    return kh_result, ps_result, acc_result , hist_text_str, hist_list


# ── Step 4: Judge prompt ──────────────────────────────────────

async def run_judge(
    job_desc: str,
    kh_result: dict,
    ps_result: dict,
    acc_result: dict,
    history_context: str
) -> dict:
    prompt = get_judge_prompt(
        job_desc,
        json.dumps(kh_result, indent=2),
        json.dumps(ps_result, indent=2),
        json.dumps(acc_result, indent=2),
        history_context
    )

    logger.info(f"the judge prompt is {prompt}")
    text = await converse_model(SONNET, prompt,tool_config=JUDGE_TOOL)
    return extract_json(text)


# ── Step 5: Store ─────────────────────────────────────────────

async def store_result(result: dict, job_desc: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        job_embedding = await get_embedding(job_desc)

        document = {
            **result,
            "timestamp": datetime.utcnow().isoformat(),
            "Job Description": job_desc
        }
        if job_embedding:
            document["job_description_vector"] = job_embedding
            logger.info("Stored with embedding")
        else:
            logger.warning("Stored without embedding")

        await loop.run_in_executor(
            None,
            lambda: opensearch.index(index=INDEX_NAME, body=document)
        )
    except Exception as e:
        logger.error(f"Store error: {e}")


# ── Main pipeline ─────────────────────────────────────────────

async def pipeline(job_desc: str) -> dict:
    # Step 1 + Step 2 in parallel (summarise and retrieve simultaneously)
    #summary, (hist_text, hist_data) = await asyncio.gather(
    #     summarise_job(job_desc),
    #     retrieve_history_wrapper(job_desc)  # wrapper defined below
    # )
    summary = await summarise_job(job_desc)

    logger.info(f"Summary: {summary}")
    # logger.info(f"job_desc: {job_desc}")
    # logger.info(f"History entries retrieved: {len(hist_data)}")

    # Step 3: Evaluate all three dimensions in parallel
    hist_text = ''
    kh_result, ps_result, acc_result , hist_text, hist_list = await evaluate_dimensions_parallel(
        summary ,job_desc
    )

    # Step 4: Judge consolidates and adjusts based on history
    final_result = await run_judge(
        summary, kh_result, ps_result, acc_result, hist_text
    )
    logger.info(f"final result")

    # Attach history rows to response
    final_result["history"] = build_history_rows(hist_list)

    logger.info(f"Final result history: {final_result}")
    return final_result


async def retrieve_history_wrapper(job_desc: str) -> tuple[str, list]:
    """
    Thin wrapper so we can run summarise + retrieve concurrently.
    Uses a lightweight Haiku call to get a quick summary for retrieval
    without waiting for the full summarise_job result.
    """

    QUICK_EXTRACT_METHODOLOGY = """You are a job analysis expert. Extract structured information from the job description below for search, classification, and retrieval purposes.

Extract the following fields:

1. job_title: The exact job title as stated in the job description.

2. summary:  suitable for Korn Ferry evaluation.

Focus on extracting and summarizing:
- Key responsibilities and accountabilities
- Required qualifications and experience
- Leadership scope and team management
- Decision-making authority and strategic impact
- Stakeholder management requirements
- Technical skills and competencies

5. job_family: The broad job family or functional category the role belongs to. """

    quick_prompt = f"""{QUICK_EXTRACT_METHODOLOGY}

Job Description:
{job_desc}

Respond with ONLY this JSON (no other text):
{{
    "job_title": "exact job title from the description",
    "summary": "summary 200-300 words atleast with all criteria",
    "job_family": "one of the job family categories listed above",
    "Reports to": "reporting officer"
}}"""
    text = await invoke_model(HAIKU, quick_prompt)
    quick_summary = extract_json(text)
    logger.info(f"quick summary {quick_summary}")
    return quick_summary
    #return await retrieve_history(quick_summary)


# ── Lambda handler ────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        job_desc_base64 = body.get("job_desc_base64")
        if not job_desc_base64:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "job_desc_base64 is required"})
            }

        decoded = base64.b64decode(job_desc_base64).decode("utf-8")
        decoded = urllib.parse.unquote(decoded)
        job_description = decoded.strip().strip('"')

        result = asyncio.run(pipeline(job_description))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": 200,
                "assessment": base64.b64encode(
                    json.dumps(result).encode()
                ).decode()
            })
        }

    except Exception as e:
        logger.error(f"Lambda error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }