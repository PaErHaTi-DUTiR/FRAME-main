from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .wikiann_loader import bio_tags_to_entities


class MasakhaNERLoader:
    """Loader for MasakhaNER 2.0 dataset.

    JSON format: {"text_list": [...], "label_list": [...], "trans_text": "..."}
    Label schema: PER, ORG, LOC, DATE (BIO tags: B-PER, I-PER, B-LOC, ...)
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)

    def available_languages(self) -> List[str]:
        langs = []
        for p in sorted(self.dataset_root.iterdir()):
            if p.is_dir() and (p / "test.json").exists():
                langs.append(p.name)
        return langs

    def load_split(self, language: str, split: str, max_samples: int = -1) -> List[Dict]:
        path = self.dataset_root / language / f"{split}.json"
        if not path.exists():
            return []

        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if max_samples > 0:
            rows = rows[:max_samples]

        out: List[Dict] = []
        for i, row in enumerate(rows):
            tokens = row.get("text_list", [])
            labels = row.get("label_list", [])
            if len(tokens) != len(labels):
                continue
            out.append({
                "id": f"masakhaner-{language}-{split}-{i:06d}",
                "dataset": "masakhaner",
                "language": language,
                "split": split,
                "tokens": tokens,
                "text": " ".join(tokens),
                "entities": bio_tags_to_entities(tokens, labels),
                "trans_text": row.get("trans_text", ""),
            })
        return out
