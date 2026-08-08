from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from model.backbones.base_backend import BaseLLMBackend
from model.prompts.span_prompts import build_span_prompt
from utils.json_parser import parse_json


@dataclass
class SpanMention:
    mention: str
    start: int
    end: int
    type_guess: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            "mention": self.mention,
            "start": self.start,
            "end": self.end,
            "type_guess": self.type_guess,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class SpanAgent:
    """Pure span detection - focuses on WHERE, delegates type to TypeAgent."""

    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        label_schema: Optional[List[str]] = None,
    ):
        self.backend = backend
        self.label_schema = [x.upper() for x in (label_schema or ["PER", "ORG", "LOC"])]
        self._noise_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of", "for",
            "is", "was", "are", "were", "be", "been", "has", "have", "had",
            "this", "that", "these", "those", "it", "its", "he", "she", "they",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        }

    def extract_spans(self, text: str, language: str, demo_examples: Optional[List[Dict]] = None) -> List[SpanMention]:
        if not self.backend:
            return []

        prompt = build_span_prompt(text=text, language=language, label_schema=self.label_schema, demo_examples=demo_examples)
        result = self.backend.generate(prompt)
        payload = parse_json(result.get("text", ""), default={})

        tokens = text.split()
        n = len(tokens)
        spans: List[SpanMention] = []
        seen: set = set()

        for row in (payload.get("spans", []) if isinstance(payload, dict) else []):
            try:
                mention = str(row.get("mention", "")).strip()
                start = int(row.get("start", -1))
                end = int(row.get("end", -1))
                type_guess = str(row.get("type_guess", self.label_schema[0])).strip().upper()
                confidence = float(row.get("confidence", 0.5))
                reason = str(row.get("reason", "span_agent"))

                if not mention or start < 0 or end <= start:
                    continue

                # Repair if out of bounds
                if end > n:
                    end = n
                if start >= n:
                    continue

                # Try to align mention text to tokens
                repaired = self._repair_span(mention, start, end, tokens)
                if repaired is None:
                    continue
                start, end = repaired

                # Trim trailing punctuation tokens
                while end > start and not self._strip_token(tokens[end - 1]):
                    end -= 1
                if end <= start:
                    continue

                span_text = " ".join(tokens[start:end])
                if self._is_noise(span_text):
                    continue

                if type_guess not in self.label_schema:
                    type_guess = self.label_schema[0]

                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)

                spans.append(SpanMention(
                    mention=span_text,
                    start=start,
                    end=end,
                    type_guess=type_guess,
                    confidence=max(0.0, min(1.0, confidence)),
                    reason=reason,
                ))
            except Exception:
                continue

        return sorted(spans, key=lambda s: (s.start, s.end))

    def _repair_span(self, mention: str, start: int, end: int, tokens: List[str]) -> Optional[tuple]:
        n = len(tokens)
        mention_norm = self._normalize_phrase(mention)
        if not mention_norm:
            return None
        span_len = len(mention_norm)

        def match(i: int, j: int) -> bool:
            if i < 0 or j > n or j <= i:
                return False
            return self._normalize_phrase(" ".join(tokens[i:j])) == mention_norm

        if match(start, end):
            return start, end

        for offset in range(0, 4):
            for s in [start - offset, start + offset]:
                if match(s, s + span_len):
                    return s, s + span_len

        for i in range(n - span_len + 1):
            if match(i, i + span_len):
                return i, i + span_len

        return None

    def _normalize_phrase(self, text: str) -> List[str]:
        parts = [re.sub(r"(^[^\w]+|[^\w]+$)", "", t).lower() for t in text.split()]
        return [p for p in parts if p]

    def _strip_token(self, token: str) -> str:
        return re.sub(r"(^[^\w]+|[^\w]+$)", "", str(token))

    def _is_noise(self, mention: str) -> bool:
        words = self._normalize_phrase(mention)
        if not words:
            return True
        if all(not any(ch.isalpha() for ch in w) for w in words):
            return True
        if len(words) == 1 and words[0] in self._noise_words:
            return True
        if len(words) == 1 and len(words[0]) <= 2 and words[0].isalpha():
            return True
        return False
