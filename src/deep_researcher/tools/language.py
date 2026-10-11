# Copyright (c) 2026 C10 Hackathon G1
"""Language gate: only articles in the allowed languages (default English) become evidence.

``RETRIEVAL_LANGUAGES`` is a comma-separated list of ISO 639-1 codes (``en``, ``de``, ``fr`` ...).
Sources that support a language parameter (newsdata.io, GDELT, Wikipedia) are asked for those
languages up front; everything else is checked after retrieval with ``langdetect`` (seeded, so
verdicts are deterministic). Text too short to classify reliably is let through as "unknown".
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ("en",)
MIN_CHARS = 40          # langdetect is unreliable below this (e.g. "x" -> Somali)

# ISO 639-1 -> the names GDELT's `sourcelang` and newsdata.io's `language` field use.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "english", "de": "german", "fr": "french", "es": "spanish", "it": "italian", "pt": "portuguese",
    "nl": "dutch", "ru": "russian", "ja": "japanese", "zh": "chinese", "ko": "korean", "ar": "arabic",
    "hi": "hindi", "sv": "swedish", "no": "norwegian", "da": "danish", "fi": "finnish", "pl": "polish",
    "tr": "turkish", "id": "indonesian",
}
NAME_TO_CODE = {v: k for k, v in LANGUAGE_NAMES.items()}


def allowed_languages() -> list[str]:
    raw = os.getenv("RETRIEVAL_LANGUAGES") or ",".join(DEFAULT_LANGUAGES)
    codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
    return codes or list(DEFAULT_LANGUAGES)


def primary_language() -> str:
    return allowed_languages()[0]


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code.lower())


def normalise_code(value: str | None) -> str | None:
    """'english' -> 'en', 'zh-cn' -> 'zh', 'EN' -> 'en'; None/'' -> None."""
    if not value:
        return None
    v = value.strip().lower()
    v = NAME_TO_CODE.get(v, v)
    return v.split("-", 1)[0].split("_", 1)[0] or None


def detect(text: str) -> str | None:
    """ISO 639-1 code of ``text`` or None when it is too short / undecidable."""
    sample = " ".join((text or "").split())[:2000]
    if len(sample) < MIN_CHARS:
        return None
    try:
        from langdetect import DetectorFactory, LangDetectException
        from langdetect import detect as _detect

        DetectorFactory.seed = 0
        return normalise_code(_detect(sample))
    except ImportError:  # pragma: no cover
        log.warning("langdetect not installed; language gate disabled")
        return None
    except LangDetectException:
        return None


def classify(text: str, declared: str | None = None,
             allowed: list[str] | None = None) -> tuple[bool, str | None]:
    """(allowed?, language). A source's own metadata wins when present; otherwise detect from text.

    Unknown language (short text, detection failure) is allowed: dropping it would lose titles-only
    results such as GDELT's, and the validator still grades relevance afterwards.
    """
    allowed = allowed or allowed_languages()
    lang = normalise_code(declared) or detect(text)
    if lang is None:
        return True, None
    return lang in allowed, lang


def info() -> dict[str, Any]:
    return {"allowed": allowed_languages(), "min_chars": MIN_CHARS}
