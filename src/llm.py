"""Thin Groq API wrapper (free-tier LLM inference)."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _client():
    from groq import Groq
    key = config.groq_api_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml "
            "(GROQ_API_KEY = \"...\") or set it as an environment variable.")
    return Groq(api_key=key)


def available() -> bool:
    return config.groq_api_key() is not None


def chat(messages, model=None, temperature=None, max_tokens=None,
         json_mode: bool = False) -> str:
    kwargs = dict(
        model=model or config.GROQ_MODEL,
        messages=messages,
        temperature=config.GROQ_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _parse_json(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"[\{\[].*[\}\]]", raw, re.S)   # grab first JSON-ish block
        if m:
            return json.loads(m.group(0))
        raise


def chat_json(messages, **kw):
    return _parse_json(chat(messages, json_mode=True, **kw))