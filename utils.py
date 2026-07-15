# ============================================================
# UTILITIES
# ============================================================

import json
import logging
import math
from typing import List, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import re

def extract_json(text: str) -> dict:
    try:
        text = text.strip()

        if not text:
            raise ValueError("Empty response from model")

        if "```json" in text:
            start = text.find("```json") + 7
            end   = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text  = parts[1].strip()
                lines = text.split('\n')
                if lines[0].strip() in ['json', 'JSON']:
                    text = '\n'.join(lines[1:])

        if not text.startswith('{') and '{' in text:
            start       = text.find('{')
            brace_count = 0
            end         = start
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            text = text[start:end]

        # -------------------------------------------------------
        # Sanitise control characters inside JSON string values.
        # Literal newlines, tabs, and carriage returns inside
        # string values are invalid JSON — replace them with
        # their escaped equivalents.
        # -------------------------------------------------------
        def sanitise_string_values(json_text: str) -> str:
            result = []
            in_string = False
            escape_next = False

            for char in json_text:
                if escape_next:
                    result.append(char)
                    escape_next = False
                    continue

                if char == '\\' and in_string:
                    result.append(char)
                    escape_next = True
                    continue

                if char == '"':
                    in_string = not in_string
                    result.append(char)
                    continue

                if in_string:
                    if char == '\n':
                        result.append('\\n')
                    elif char == '\r':
                        result.append('\\r')
                    elif char == '\t':
                        result.append('\\t')
                    elif ord(char) < 32:
                        # Replace any other control characters
                        result.append(f'\\u{ord(char):04x}')
                    else:
                        result.append(char)
                else:
                    result.append(char)

            return ''.join(result)

        text = sanitise_string_values(text)

        return json.loads(text)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}\nProblematic text: {text}")
        return {"error": "Failed to parse model response", "raw": text}
    except Exception as e:
        logger.error(f"Unexpected error in extract_json: {e}")
        return {"error": str(e)}


# def extract_json(text: str) -> dict:
#     try:
#         text = text.strip()

#         if not text:
#             raise ValueError("Empty response from model")

#         # Strip markdown code fences
#         if "```json" in text:
#             start = text.find("```json") + 7
#             end = text.find("```", start)
#             if end != -1:
#                 text = text[start:end].strip()
#         elif "```" in text:
#             parts = text.split("```")
#             if len(parts) >= 3:
#                 text = parts[1].strip()
#                 lines = text.split('\n')
#                 if lines[0].strip().lower() == 'json':
#                     text = '\n'.join(lines[1:])

#         # Extract JSON object if mixed with prose
#         if not text.startswith('{') and '{' in text:
#             start = text.find('{')
#             brace_count = 0
#             end = start
#             for i, char in enumerate(text[start:], start):
#                 if char == '{':
#                     brace_count += 1
#                 elif char == '}':
#                     brace_count -= 1
#                     if brace_count == 0:
#                         end = i + 1
#                         break
#             text = text[start:end]

#         return json.loads(text)

#     except json.JSONDecodeError as e:
#         logger.error(f"JSON decode error: {e} | Text: {text[:300]}")
#         return _default_error_response("Failed to parse model response")
#     except Exception as e:
#         logger.error(f"Unexpected error in extract_json: {e}")
#         return _default_error_response("Error processing model response")


def _default_error_response(message: str) -> dict:
    return {
        "job_title": "Unknown",
        "summary": message,
        "key_skills": [],
        "seniority": "Unknown",
        "job_family": "Unknown",
        "comments": message,
        "know_how_score": 0
    }


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def build_history_text(hist: list) -> tuple[str, list]:
    """
    Parse OpenSearch hits into a readable text block and structured data list.
    Returns (hist_text, hist_data).
    """
    hist_text = ""
    hist_data = []

    for h in hist:
        if "_source" not in h:
            continue

        s = h["_source"]
        score = h.get("_score", 0)

        know_how = s.get('Know-How', 'N/A')
        if len(str(know_how)) > 500:
            know_how = str(know_how)[:500] + "..."

        hist_text += f"""
Title: {s.get('Job Title', 'N/A')}
Grade: {s.get('Approved grade', 'N/A')}
KnowHow: {know_how}
KH1: {s.get('KH1', 'N/A')} | KH2: {s.get('KH2', 'N/A')} | KH3: {s.get('KH3', 'N/A')}
KH Points: {s.get('KH Point', 'N/A')}
PS TE: {s.get('PS1', 'N/A')} | PS TC: {s.get('PS2', 'N/A')}
ACC FTA: {s.get('ACC1', 'N/A')} | ACC AOI: {s.get('ACC2', 'N/A')} | ACC NOI: {s.get('ACC3', 'N/A')}
Total Points: {s.get('Grand Total', 'N/A')}
Similarity Score: {score:.3f}
Description: {s.get('Job Description', 'N/A')[:200]}...
"""

        hist_data.append({
            "title": s.get('Job Title', 'N/A'),
            "profile": s.get('Profile', 'N/A'),
            "kh_points": f"{s.get('KH1', 'N/A')} {s.get('KH2', 'N/A')} {s.get('KH3', 'N/A')} ({s.get('KH Point', 'N/A')} pts)",
            "ps_points": f"{s.get('PS1', 'N/A')} {s.get('PS2', 'N/A')} {s.get('PS3', 'N/A')} ({s.get('PS Score', 'N/A')} pts)",
            "acc_points": f"{s.get('ACC1', 'N/A')} {s.get('ACC2', 'N/A')} {s.get('ACC3', 'N/A')} ({s.get('ACC Point', 'N/A')} pts)",
            "total_points": s.get('Grand Total', 'N/A'),
        })

    return hist_text[:2000], hist_data


def build_history_rows(hist_data: list) -> dict:
    """Format hist_data into the final history payload for the response."""
    if not hist_data:
        return {"rows": [], "total_retrieved": 0, "retrieval_method": "none"}

    logger.info(f"the hist text is {hist_data}")
    if isinstance(hist_data, str):
        logger.error(f"build_history_rows received a string instead of a list: {hist_data}")
        return {"rows": [], "total_retrieved": 0, "retrieval_method": "error"}

    if not isinstance(hist_data, list):
        logger.error(f"build_history_rows received unexpected type: {type(hist_data)}")
        return {"rows": [], "total_retrieved": 0, "retrieval_method": "error"}


    rows = [
        {
            "JobId": item.get('title', f'Job_{i+1}'),
            "KnowHow": item.get('kh_points', 'N/A'),
            "probSlv": item.get('ps_points', 'N/A'),
            "acc": item.get('acc_points', 'N/A'),
            "hayPts": str(item.get('total_points', 'N/A')),
            "profile": item.get('profile', 'N/A'),
        }
        for i, item in enumerate(hist_data[:2])
    ]

    return {
        "rows": rows,
        "total_retrieved": len(hist_data),
        "retrieval_method": "vector_similarity"
    }