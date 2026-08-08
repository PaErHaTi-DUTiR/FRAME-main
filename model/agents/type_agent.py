from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from model.backbones.base_backend import BaseLLMBackend
from model.prompts.type_prompts import build_type_prompt
from utils.json_parser import parse_json


@dataclass
class TypeClassification:
    span_id: int
    entity_type: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            "id": self.span_id,
            "type": self.entity_type,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class TypeAgent:
    """Type classification agent — uses English translation to determine entity types."""

    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        label_schema: Optional[List[str]] = None,
        cot_mode: bool = False,
    ):
        self.backend = backend
        self.label_schema = [x.upper() for x in (label_schema or ["PER", "ORG", "LOC"])]
        self.cot_mode = cot_mode

    def classify(
        self,
        trans_text: str,
        mentions: List[Dict],
        language: str,
    ) -> List[TypeClassification]:
        """Classify each mention's entity type using English translation context.

        Args:
            trans_text: English translation of the original sentence.
            mentions: List of {"id": int, "mention": str, "type_guess": str}.
            language: Original language code.

        Returns:
            List of TypeClassification, one per mention id that was classified.
        """
        if not self.backend or not mentions or not trans_text.strip():
            return []

        prompt = build_type_prompt(
            trans_text=trans_text,
            mentions=mentions,
            language=language,
            label_schema=self.label_schema,
            use_cot=self.cot_mode,
        )
        result = self.backend.generate(prompt)
        payload = parse_json(result.get("text", ""), default={})

        # NONE/O are valid rejection signals from TypeAgent
        valid_types = set(self.label_schema) | {"NONE", "O"}
        classifications: List[TypeClassification] = []

        for row in (payload.get("classifications", []) if isinstance(payload, dict) else []):
            try:
                span_id = int(row.get("id", -1))
                if span_id < 0:
                    continue
                entity_type = str(row.get("type", "")).strip().upper()
                # Keep NONE/O as-is; map unknown types to first schema type
                if entity_type not in valid_types:
                    entity_type = self.label_schema[0]
                confidence = float(row.get("confidence", 0.5))
                reason = str(row.get("reason", "type_agent"))
                classifications.append(TypeClassification(
                    span_id=span_id,
                    entity_type=entity_type,
                    confidence=max(0.0, min(1.0, confidence)),
                    reason=reason,
                ))
            except Exception:
                continue

        return classifications
