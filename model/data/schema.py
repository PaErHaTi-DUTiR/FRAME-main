from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class EntitySpan:
    start: int
    end: int
    text: str
    label: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class NERSample:
    sample_id: str
    dataset: str
    language: str
    split: str
    tokens: List[str]
    text: str
    entities: List[EntitySpan]

    def to_dict(self) -> Dict:
        return {
            "id": self.sample_id,
            "dataset": self.dataset,
            "language": self.language,
            "split": self.split,
            "tokens": self.tokens,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
        }
