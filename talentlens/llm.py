"""Thin wrapper around the Claude API used by extraction, JD parsing and scoring."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import anthropic
from pydantic import BaseModel


def _load_dotenv(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()

MODEL = os.getenv("TALENTLENS_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("TALENTLENS_EFFORT", "medium")  # low | medium | high
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens (Claude Opus 5.5 list price) - used for cost reporting only.
PRICE_IN, PRICE_OUT = 4.00, 20.00

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model cannot return a usable answer."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1e6 * PRICE_IN + self.output_tokens / 1e6 * PRICE_OUT


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (or an `ant auth login` profile)
    return _client


def api_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def image_block(media_type: str, data: bytes) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(data).decode()},
    }


def structured_call(system: str, content: list[dict], output_model: type[T], max_tokens: int = 16000) -> tuple[T, Usage]:
    """One Claude call whose answer is validated against a Pydantic model.

    Server-side fallbacks are on: if the model declines on a safety classifier,
    the API retries on a fallback model inside the same call.
    """
    try:
        resp = get_client().beta.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=output_model,
            output_config={"effort": EFFORT},
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("API key ไม่ถูกต้อง (authentication failed)") from e
    except anthropic.RateLimitError as e:
        raise LLMError("ถูกจำกัดอัตราการเรียก API (rate limited) - ลองใหม่อีกครั้ง") from e
    except anthropic.BadRequestError as e:
        raise LLMError(f"คำขอไม่ถูกต้อง: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("เชื่อมต่อ API ไม่ได้ - ตรวจสอบอินเทอร์เน็ต") from e

    usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
    if resp.stop_reason == "refusal":
        raise LLMError("โมเดลปฏิเสธคำขอนี้ (refusal)")
    if resp.stop_reason == "max_tokens":
        raise LLMError("คำตอบยาวเกิน max_tokens")
    if resp.parsed_output is None:
        raise LLMError("โมเดลไม่ได้ส่งข้อมูลตาม schema")
    return resp.parsed_output, usage
