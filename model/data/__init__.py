from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from utils.common import read_jsonl

from .masakhaner_loader import MasakhaNERLoader
from .quality_sampler import hard_filter_records, is_clean_entity, sample_records_with_quality_distribution
from .sampler import low_resource_sample
from .schema import EntitySpan, NERSample
from .wikiann_loader import WikiANNLoader, bio_tags_to_entities, iter_conll_sentences


def discover_languages(dataset_root: Path, marker_file: str) -> List[str]:
    langs = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and (p / marker_file).exists():
            langs.append(p.name)
    return langs


def prepare_wikiann_split(dataset_root: Path, language: str, split: str, max_samples: int = -1) -> List[Dict]:
    return WikiANNLoader(dataset_root).load_split(language, split, max_samples=max_samples)


def prepare_masakhaner_split(dataset_root: Path, language: str, split: str, max_samples: int = -1) -> List[Dict]:
    return MasakhaNERLoader(dataset_root).load_split(language, split, max_samples=max_samples)


def load_processed_split(
    processed_root: Path,
    dataset_name: str,
    language: str,
    split: str,
    max_samples: int | None = None,
    low_resource_ratio: float = 1.0,
    seed: int = 42,
    quality_sample: bool = False,
) -> List[Dict]:
    path = processed_root / dataset_name / language / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")

    records = read_jsonl(path)

    if quality_sample and max_samples and max_samples > 0:
        # Step 1: hard-filter annotation artifacts (bracket disambiguations, etc.)
        records = hard_filter_records(records)
        # Step 2: stratified quality sampling to get exactly max_samples clean records
        records, _ = sample_records_with_quality_distribution(
            records, target_size=max_samples, seed=seed,
            sampling_config={"profile": "balanced", "preserve_label_distribution": True},
        )
    else:
        if max_samples is not None and max_samples > 0:
            records = records[:max_samples]
        if low_resource_ratio < 1.0:
            records = low_resource_sample(records, low_resource_ratio, seed=seed)

    return records


__all__ = [
    "EntitySpan",
    "NERSample",
    "WikiANNLoader",
    "MasakhaNERLoader",
    "bio_tags_to_entities",
    "iter_conll_sentences",
    "discover_languages",
    "prepare_wikiann_split",
    "prepare_masakhaner_split",
    "sample_records_with_quality_distribution",
    "low_resource_sample",
    "load_processed_split",
]
