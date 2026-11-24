"""Free backend: an open-source model served by Ollama (local PC or a Colab GPU).

Same interface as ``llm.structured_call`` so extraction, JD parsing and scoring
can switch backends without changing their prompts.

    ollama serve &            # starts the server on http://localhost:11434
    ollama pull qwen2.5:7b    # ~4.7 GB, fits a free Colab T4 GPU
"""
from __future__ import annotations

import json
import os
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .llm import LLMError, Usage

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("TALENTLENS_LOCAL_MODEL", "qwen2.5:7b")
NUM_CTX = int(os.getenv("TALENTLENS_NUM_CTX", "8192"))  # Thai text uses many tokens; default 2-4k would truncate CVs
TIMEOUT = float(os.getenv("TALENTLENS_LOCAL_TIMEOUT", "600"))

T = TypeVar("T", bound=BaseModel)


def available() -> bool:
    try:
        tags = httpx.get(f"{HOST}/api/tags", timeout=3).json()
        return any(m.get("name", "").startswith(MODEL.split(":")[0]) for m in tags.get("models", []))
    except Exception:
        return False


def _chat(messages: list[dict], fmt) -> dict:
    body = {"model": MODEL, "messages": messages, "stream": False, "format": fmt,
            "options": {"temperature": 0, "num_ctx": NUM_CTX}}
    try:
        r = httpx.post(f"{HOST}/api/chat", json=body, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise LLMError(f"ติดต่อ Ollama ที่ {HOST} ไม่ได้ - รัน `ollama serve` แล้วหรือยัง? ({e})") from e
    if r.status_code == 404:
        raise LLMError(f"ไม่พบโมเดล {MODEL} - รัน `ollama pull {MODEL}` ก่อน")
    if r.status_code >= 400:
        raise LLMError(f"Ollama error {r.status_code}: {r.text[:200]}")
    return r.json()


def structured_call(system: str, content: list[dict], output_model: type[T], max_tokens: int = 16000) -> tuple[T, Usage]:
    """Ask the local model for JSON matching ``output_model``; retry once if it does not validate."""
    text = "\n\n".join(b["text"] for b in content if b.get("type") == "text")
    if any(b.get("type") == "image" for b in content):
        raise LLMError("โมเดลในเครื่องรับเฉพาะข้อความ - ต้อง OCR ภาพก่อน")
    schema = output_model.model_json_schema()
    messages = [
        {"role": "system", "content": system + "\n\nAnswer with one JSON object that follows this JSON schema exactly:\n"
                                       + json.dumps(schema, ensure_ascii=False)},
        {"role": "user", "content": text},
    ]
    usage = Usage()
    last_error = ""
    for attempt in range(2):
        try:
            resp = _chat(messages, schema)
        except LLMError:
            if attempt:
                raise
            resp = _chat(messages, "json")  # older Ollama versions only accept "json"
        usage.add(Usage(resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)))
        answer = resp.get("message", {}).get("content", "")
        try:
            return output_model.model_validate_json(answer), usage
        except ValidationError as e:
            last_error = str(e)[:500]
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": f"That JSON did not match the schema: {last_error}\nReturn the corrected JSON only."}]
    raise LLMError(f"โมเดลในเครื่องตอบ JSON ไม่ตรง schema: {last_error[:200]}")
