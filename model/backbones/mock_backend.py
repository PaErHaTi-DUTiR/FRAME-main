from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Dict, List

from .base_backend import BaseLLMBackend, ChatMessage


class MockLLMBackend(BaseLLMBackend):
    """Deterministic mock backend for end-to-end pipeline testing."""

    def __init__(
        self,
        model_name: str = "mock-llm",
        deterministic: bool = True,
        seed: int = 42,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.deterministic = deterministic
        self.seed = seed

    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict:
        prompt = messages[-1].content if messages else ""
        payload = self._dispatch(prompt=prompt)
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "raw": {
                "backend": "mock",
                "deterministic": self.deterministic,
                "model_name": self.model_name,
            },
        }

    def _dispatch(self, prompt: str) -> Dict:
        if "You are Candidate Agent in KARVE." in prompt:
            return self._candidate_payload(prompt)
        if "You are Retrieval Agent in KARVE." in prompt:
            return self._retrieval_payload(prompt)
        if "You are Alignment Agent in KARVE." in prompt:
            return self._alignment_payload(prompt)
        if "You are Verification Agent in KARVE." in prompt:
            return self._verification_payload(prompt)
        if "You are Reflection Agent in KARVE." in prompt:
            return self._reflection_payload(prompt)
        return {"message": "mock_default"}

    def _candidate_payload(self, prompt: str) -> Dict:
        text = self._extract_field(prompt, "Text")
        tokens = text.split()
        candidates = []

        spans = []
        idx = 0
        while idx < len(tokens):
            tok = tokens[idx]
            if tok and tok[0].isupper():
                j = idx + 1
                while j < len(tokens) and tokens[j] and tokens[j][0].isupper():
                    j += 1
                spans.append((idx, j, " ".join(tokens[idx:j])))
                idx = j
            else:
                idx += 1

        rng = self._rng(prompt)
        if not spans and tokens:
            pos = min(len(tokens) - 1, 0)
            spans = [(pos, pos + 1, tokens[pos])]

        for start, end, mention in spans[:3]:
            conf = round(0.6 + 0.2 * rng.random(), 3)
            candidates.append(
                {
                    "mention": mention,
                    "start": start,
                    "end": end,
                    "type": "MISC",
                    "confidence": conf,
                    "reason": "mock_candidate_high_recall",
                }
            )

        return {"candidates": candidates}

    def _retrieval_payload(self, prompt: str) -> Dict:
        mention = self._extract_field(prompt, "Mention") or "mock_mention"
        aliases = [mention, mention.lower()]
        return {
            "query": mention,
            "aliases": aliases,
            "reason": "mock_query_rewrite",
        }

    def _alignment_payload(self, prompt: str) -> Dict:
        evidence_json = self._extract_field(prompt, "Evidences")
        best_id = ""
        best_title = ""
        if evidence_json:
            try:
                rows = json.loads(evidence_json)
                if isinstance(rows, list) and rows:
                    first = rows[0]
                    if isinstance(first, dict):
                        best_id = str(first.get("evidence_id", ""))
                        best_title = str(first.get("title", ""))
            except Exception:
                pass

        return {
            "best_evidence_id": best_id,
            "best_title": best_title,
            "alignment_score": 0.76,
            "status": "aligned",
            "signal_scores": {
                "alias": 0.8,
                "semantic": 0.7,
                "language_bridge": 0.8,
                "source": 0.7,
            },
            "reason": "mock_alignment_decision",
        }

    def _verification_payload(self, prompt: str) -> Dict:
        _ = prompt
        return {
            "support_score": 0.74,
            "type_consistency": 0.62,
            "conflict_score": 0.21,
            "decision": "accept",
            "decision_reason": "mock_evidence_grounded_verification",
        }

    def _reflection_payload(self, prompt: str) -> Dict:
        _ = prompt
        return {
            "keep_entity": True,
            "revised_type": "MISC",
            "confidence": 0.71,
            "revision_action": "keep",
            "reason": "mock_reflection_result",
        }

    def _extract_field(self, prompt: str, field: str) -> str:
        pattern = rf"{re.escape(field)}:\\s*(.*)"
        match = re.search(pattern, prompt)
        return match.group(1).strip() if match else ""

    def _rng(self, prompt: str) -> random.Random:
        if not self.deterministic:
            return random.Random()

        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16)
        return random.Random(self.seed + offset)
