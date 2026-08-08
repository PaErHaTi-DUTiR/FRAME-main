from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def iter_conll_sentences(path: Path) -> Iterable[Tuple[List[str], List[str]]]:
    tokens: List[str] = []
    tags: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    yield tokens, tags
                    tokens, tags = [], []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            token, tag = parts[0], parts[-1]
            tokens.append(token)
            tags.append(tag)

    if tokens:
        yield tokens, tags


def bio_tags_to_entities(tokens: Sequence[str], tags: Sequence[str]) -> List[Dict]:
    entities: List[Dict] = []
    start = None
    cur_label = None

    def flush(end_idx: int) -> None:
        nonlocal start, cur_label
        if start is None or cur_label is None:
            return
        entities.append(
            {
                "start": start,
                "end": end_idx,
                "text": " ".join(tokens[start:end_idx]),
                "label": cur_label,
            }
        )
        start = None
        cur_label = None

    for i, raw_tag in enumerate(tags):
        if raw_tag == "O":
            flush(i)
            continue
        prefix, label = raw_tag.split("-", 1) if "-" in raw_tag else ("B", raw_tag)
        if prefix == "B":
            flush(i)
            start = i
            cur_label = label
        elif prefix == "I":
            if start is None or cur_label != label:
                flush(i)
                start = i
                cur_label = label
        else:
            flush(i)

    flush(len(tags))
    return entities


class WikiANNLoader:
    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)

    def available_languages(self) -> List[str]:
        langs = []
        for p in sorted(self.dataset_root.iterdir()):
            if p.is_dir() and (p / "train.txt").exists():
                langs.append(p.name)
        return langs

    def load_split(self, language: str, split: str, max_samples: int = -1) -> List[Dict]:
        json_path = self.dataset_root / language / f"{split}.json"
        if json_path.exists():
            return self._load_json_split(language=language, split=split, path=json_path, max_samples=max_samples)

        txt_path = self.dataset_root / language / f"{split}.txt"
        if not txt_path.exists():
            return []
        return self._load_txt_split(language=language, split=split, path=txt_path, max_samples=max_samples)

    def _load_json_split(self, language: str, split: str, path: Path, max_samples: int = -1) -> List[Dict]:
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                tokens = row.get("text_list", [])
                tags = row.get("label_list", [])
                if len(tokens) != len(tags):
                    continue
                trans_text = str(row.get("trans_text", "")).strip()
                record: Dict = {
                    "id": f"wikiann-{language}-{split}-{i:06d}",
                    "dataset": "wikiann",
                    "language": language,
                    "split": split,
                    "tokens": tokens,
                    "text": " ".join(tokens),
                    "entities": bio_tags_to_entities(tokens, tags),
                }
                if trans_text:
                    record["trans_text"] = trans_text
                records.append(record)
        return records

    def _load_txt_split(self, language: str, split: str, path: Path, max_samples: int = -1) -> List[Dict]:
        records: List[Dict] = []
        for i, (tokens, tags) in enumerate(iter_conll_sentences(path)):
            if max_samples > 0 and i >= max_samples:
                break
            records.append(
                {
                    "id": f"wikiann-{language}-{split}-{i:06d}",
                    "dataset": "wikiann",
                    "language": language,
                    "split": split,
                    "tokens": tokens,
                    "text": " ".join(tokens),
                    "entities": bio_tags_to_entities(tokens, tags),
                }
            )
        return records
