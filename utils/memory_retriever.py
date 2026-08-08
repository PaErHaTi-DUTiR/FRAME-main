from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import numpy as np


class DemoMemoryRetriever:
    """
    Retrieves few-shot demo examples for the CandidateAgent.

    Modes:
    - "jaccard": lexical overlap (fast, language-agnostic surface matching)
    - "dense": monolingual semantic similarity via all-MiniLM-L6-v2
    - "cross_lingual": multilingual semantic similarity via
        paraphrase-multilingual-MiniLM-L12-v2; supports retrieving English
        examples for low-resource target languages
    - "random": uniform random sample
    - "type_balanced": random but ensures each entity type is represented
    - "type_coverage": greedy selection maximizing entity type diversity across
        the k selected demos; each successive demo adds the most NEW entity types
        not yet covered; crucial for fine-grained schemas (e.g. MC2's 33 types)
    """

    MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    MONOLINGUAL_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        demo_records: List[Dict],
        k: int = 2,
        mode: str = "cross_lingual",
        aux_records: Optional[List[Dict]] = None,
        label_schema: Optional[List[str]] = None,
    ):
        """
        Args:
            demo_records: Primary pool (typically target-language training examples).
            k: Number of examples to retrieve.
            mode: Retrieval mode (see class docstring).
            aux_records: Auxiliary pool (e.g. English examples for cross-lingual mode).
                When provided and mode is 'cross_lingual', the retriever searches
                both pools and returns the top-k across both.
        """
        self.k = k
        self.mode = mode
        self.label_schema = [x.upper() for x in (label_schema or [])]
        self.all_records: List[Dict] = list(demo_records)
        if aux_records:
            self.all_records.extend(aux_records)

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        if mode in ("dense", "cross_lingual"):
            model_name = self.MULTILINGUAL_MODEL if mode == "cross_lingual" else self.MONOLINGUAL_MODEL
            try:
                import warnings
                warnings.filterwarnings("ignore")
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(model_name)
                texts = [r.get("text", "") for r in self.all_records]
                self._embeddings = self._encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                self._use_embeddings = True
            except Exception as e:
                print(f"[DemoMemoryRetriever] {mode} failed ({e}), falling back to jaccard.")
                self.mode = "jaccard"
                self._use_embeddings = False
        else:
            self._use_embeddings = False

        if self.mode == "jaccard":
            self._doc_sets = []
            for r in self.all_records:
                tokens = r.get("tokens", []) or r.get("text", "").split()
                self._doc_sets.append(set(t.lower() for t in tokens))

    def retrieve(self, text: str) -> List[Dict]:
        if not self.all_records or self.k <= 0:
            return []

        if self.mode == "random":
            return random.sample(self.all_records, min(self.k, len(self.all_records)))

        if self.mode == "type_balanced":
            return self._type_balanced_sample()

        if self.mode == "type_coverage":
            return self._type_coverage_sample()

        if self.mode == "jaccard":
            query_set = set(text.lower().split())
            scores = []
            for i, dset in enumerate(self._doc_sets):
                union = len(query_set | dset)
                score = len(query_set & dset) / union if union > 0 else 0.0
                scores.append((score, i))
            scores.sort(key=lambda x: x[0], reverse=True)
            # De-duplicate by entity content to avoid repetitive demos
            seen, results = set(), []
            for _, idx in scores:
                key = frozenset(e.get("text", e.get("mention", "")) for e in self.all_records[idx].get("entities", []))
                if key not in seen:
                    seen.add(key)
                    results.append(self.all_records[idx])
                if len(results) >= self.k:
                    break
            return results

        if self._use_embeddings:
            query_emb = self._encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
            norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
            norms[norms == 0] = 1e-10
            scores = np.dot(self._embeddings, query_emb) / norms
            top_idx = np.argsort(scores)[::-1]
            # De-duplicate and ensure entity-type diversity
            seen_keys, results = set(), []
            for idx in top_idx:
                rec = self.all_records[idx]
                key = frozenset(e.get("text", e.get("mention", "")) for e in rec.get("entities", []))
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(rec)
                if len(results) >= self.k:
                    break
            return results

        return []

    def _type_balanced_sample(self) -> List[Dict]:
        """Sample ensuring each entity type has at least one example.

        Uses label_schema if provided; falls back to PER/ORG/LOC for WikiANN schemas.
        """
        by_type: Dict[str, List[Dict]] = {}
        for r in self.all_records:
            for e in r.get("entities", []):
                lbl = e.get("label", e.get("type", ""))
                if lbl:
                    by_type.setdefault(lbl, []).append(r)
        target_types = self.label_schema if self.label_schema else ["PER", "ORG", "LOC"]
        results = []
        seen_ids = set()
        for lbl in target_types:
            candidates = [r for r in by_type.get(lbl, []) if id(r) not in seen_ids]
            if candidates:
                pick = random.choice(candidates)
                results.append(pick)
                seen_ids.add(id(pick))
            if len(results) >= self.k:
                break
        # Fill remaining slots randomly
        remaining = [r for r in self.all_records if id(r) not in seen_ids]
        if remaining and len(results) < self.k:
            results += random.sample(remaining, min(self.k - len(results), len(remaining)))
        return results[:self.k]

    def _type_coverage_sample(self) -> List[Dict]:
        """Greedy demo selection maximizing entity type coverage across all k demos.

        Selects demos so that together they cover the maximum number of distinct
        entity types. Each iteration picks the record that adds the most NEW types
        not yet seen in already-selected demos. Ideal for fine-grained schemas
        (e.g. MC2's 33 types) where cross_lingual similarity tends to over-select
        the dominant type (OtherPER) and never show rare types.
        """
        # Pre-compute entity type sets per record
        record_type_sets = []
        for r in self.all_records:
            types = frozenset(
                str(e.get("label", e.get("type", ""))).strip().upper()
                for e in r.get("entities", [])
                if e.get("label", e.get("type", ""))
            )
            record_type_sets.append(types)

        covered: set = set()
        results: List[Dict] = []
        seen_ids: set = set()

        for _ in range(self.k):
            best_idx = -1
            best_gain = -1
            for i, r in enumerate(self.all_records):
                if id(r) in seen_ids:
                    continue
                gain = len(record_type_sets[i] - covered)
                # Tie-break: prefer records with entities (non-empty type set)
                if gain > best_gain or (gain == best_gain and record_type_sets[i] and best_idx == -1):
                    best_gain = gain
                    best_idx = i

            if best_idx < 0:
                break
            results.append(self.all_records[best_idx])
            seen_ids.add(id(self.all_records[best_idx]))
            covered |= record_type_sets[best_idx]

        return results
