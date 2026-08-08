from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


_LABEL_TOKEN_PATTERN = re.compile(r"^(?:[BIOES]-[A-Za-z][A-Za-z0-9_]*|O)$")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Tokens that indicate Wikipedia-style disambiguation artifacts (e.g. "City ( State )")
_BRACKET_TOKENS = frozenset({"(", ")", "[", "]", "{", "}"})
_MAX_ENTITY_TOKENS = 6  # entities longer than this are almost always annotation artifacts


def is_clean_entity(entity_text: str) -> bool:
    """Return True if entity text looks like a genuine NER span (not a Wikipedia artifact)."""
    tokens = entity_text.split()
    if not tokens:
        return False
    # Contains standalone bracket tokens  →  Wikipedia disambiguation artifact
    if any(t in _BRACKET_TOKENS for t in tokens):
        return False
    # Any token is pure punctuation (no alphabetic/digit character)
    if any(not re.search(r"\w", t) for t in tokens):
        return False
    # Must start with an alphanumeric character
    if not re.match(r"^\w", tokens[0]):
        return False
    # Suspiciously long span
    if len(tokens) > _MAX_ENTITY_TOKENS:
        return False
    return True


def hard_filter_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove records whose gold entities contain annotation artifacts.

    Keeps records where every entity passes is_clean_entity().
    Records with zero entities are kept (they are valid O-only sentences).
    """
    cleaned = []
    for rec in records:
        entities = rec.get("entities", []) or []
        if all(is_clean_entity(str(e.get("text", ""))) for e in entities):
            cleaned.append(rec)
    return cleaned


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _quantile(sorted_values: List[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _to_probs(counter: Counter[str]) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items() if v > 0}


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-12
    labels = set(p.keys()) | set(q.keys())
    if not labels:
        return 0.0
    m = {k: 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0) for k in labels}

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        s = 0.0
        for k in labels:
            av = max(a.get(k, 0.0), eps)
            bv = max(b.get(k, 0.0), eps)
            s += av * math.log(av / bv)
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _alloc_quota(counter: Counter[str], target: int) -> Dict[str, int]:
    if target <= 0:
        return {}
    total = sum(counter.values())
    if total <= 0:
        return {}

    raw: Dict[str, float] = {k: target * (v / total) for k, v in counter.items()}
    out: Dict[str, int] = {k: int(math.floor(v)) for k, v in raw.items()}
    remain = target - sum(out.values())

    if remain > 0:
        remainders = sorted(((raw[k] - out[k], k) for k in raw.keys()), reverse=True)
        for _, key in remainders[:remain]:
            out[key] += 1
    return out


def _entity_label_counts(records: List[Dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for rec in records:
        for ent in rec.get("entities", []) or []:
            label = str(ent.get("label", "")).strip()
            if label:
                counter[label] += 1
    return counter


def _primary_label(record: Dict[str, Any]) -> str:
    labels = [
        str(e.get("label", "")).strip()
        for e in (record.get("entities", []) or [])
        if str(e.get("label", "")).strip()
    ]
    if not labels:
        return "__NO_ENTITY__"
    return Counter(labels).most_common(1)[0][0]


def _span_validity_ratio(record: Dict[str, Any], token_len: int) -> float:
    entities = record.get("entities", []) or []
    if not entities:
        return 1.0
    valid = 0
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        text = str(ent.get("text", "")).strip()
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= token_len and text:
            valid += 1
    return valid / max(1, len(entities))


def _length_bucket(token_len: int, p33: float, p66: float) -> str:
    if token_len <= p33:
        return "short"
    if token_len <= p66:
        return "medium"
    return "long"


def _build_quality_context(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_lens = [len(r.get("tokens", []) or []) for r in records]
    ent_counts = [len(r.get("entities", []) or []) for r in records]
    unique_label_counts = [
        len(
            {
                str(e.get("label", "")).strip()
                for e in (r.get("entities", []) or [])
                if str(e.get("label", "")).strip()
            }
        )
        for r in records
    ]

    sorted_lens = sorted(token_lens)
    p33 = _quantile(sorted_lens, 0.33)
    p66 = _quantile(sorted_lens, 0.66)
    med = _quantile(sorted_lens, 0.50)
    q25 = _quantile(sorted_lens, 0.25)
    q75 = _quantile(sorted_lens, 0.75)
    iqr = max(1.0, q75 - q25)

    avg_entities = sum(ent_counts) / max(1, len(ent_counts))
    max_unique_labels = max(1, max(unique_label_counts) if unique_label_counts else 1)

    return {
        "p33": p33,
        "p66": p66,
        "median_len": med,
        "iqr_len": iqr,
        "avg_entities": avg_entities,
        "max_unique_labels": max_unique_labels,
    }


def _quality_components(record: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, float]:
    tokens = [str(x) for x in (record.get("tokens", []) or [])]
    text = str(record.get("text", "") or "")
    entities = record.get("entities", []) or []

    token_len = len(tokens)
    len_score = max(0.0, 1.0 - abs(token_len - ctx["median_len"]) / (ctx["iqr_len"] + 1.0))

    if ctx["avg_entities"] > 0:
        entity_score = min(1.0, len(entities) / (ctx["avg_entities"] + 1.0))
    else:
        entity_score = 1.0 if not entities else 0.8

    label_set = {str(e.get("label", "")).strip() for e in entities if str(e.get("label", "")).strip()}
    diversity_score = min(1.0, len(label_set) / max(1, ctx["max_unique_labels"]))

    norm_tokens = [t.strip().lower() for t in tokens if t.strip()]
    if norm_tokens:
        token_counter = Counter(norm_tokens)
        dominant = token_counter.most_common(1)[0][1] / len(norm_tokens)
        uniq_ratio = len(set(norm_tokens)) / len(norm_tokens)
        semantic_score = 0.5 * uniq_ratio + 0.5 * (1.0 - dominant)
    else:
        semantic_score = 0.0

    nonempty_ratio = len(norm_tokens) / max(1, len(tokens))
    has_text = 1.0 if text.strip() else 0.0
    span_validity = _span_validity_ratio(record, token_len)
    completeness_score = 0.4 * has_text + 0.3 * nonempty_ratio + 0.3 * span_validity

    pollution_tokens = sum(1 for t in norm_tokens if _LABEL_TOKEN_PATTERN.match(t))
    pollution_ratio = pollution_tokens / max(1, len(norm_tokens))
    control_hits = len(_CONTROL_CHAR_PATTERN.findall(text))
    control_ratio = control_hits / max(1, len(text))
    replacement_penalty = 0.2 if "\ufffd" in text else 0.0
    cleanliness_score = max(0.0, 1.0 - min(1.0, 1.4 * pollution_ratio + control_ratio + replacement_penalty))

    total = (
        0.20 * len_score
        + 0.20 * entity_score
        + 0.15 * diversity_score
        + 0.15 * semantic_score
        + 0.15 * completeness_score
        + 0.15 * cleanliness_score
    )

    return {
        "length": _safe_float(len_score),
        "entity_coverage": _safe_float(entity_score),
        "diversity": _safe_float(diversity_score),
        "semantic": _safe_float(semantic_score),
        "completeness": _safe_float(completeness_score),
        "cleanliness": _safe_float(cleanliness_score),
        "total": _safe_float(total),
    }


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def sample_records_with_quality_distribution(
    records: List[Dict[str, Any]],
    target_size: int,
    seed: int = 42,
    sampling_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if target_size <= 0:
        return [], {
            "source_count": len(records),
            "selected_count": 0,
            "target_count": target_size,
            "warning": "target_size <= 0",
        }

    if not records:
        return [], {
            "source_count": 0,
            "selected_count": 0,
            "target_count": target_size,
            "warning": "empty source",
        }

    cfg = dict(sampling_config or {})
    profile = str(cfg.get("profile", "balanced")).strip().lower()
    if profile not in {"balanced", "easy"}:
        profile = "balanced"

    easy_top_fraction = max(0.0, min(1.0, float(cfg.get("easy_top_fraction", 0.75))))
    easy_min_quality = max(0.0, min(1.0, float(cfg.get("easy_min_quality", 0.0))))

    easy_max_entities_raw = cfg.get("easy_max_entities")
    easy_max_entities: Optional[int]
    if easy_max_entities_raw is None:
        easy_max_entities = None
    else:
        try:
            easy_max_entities = int(easy_max_entities_raw)
        except Exception:
            easy_max_entities = None
        if easy_max_entities is not None and easy_max_entities <= 0:
            easy_max_entities = None

    prefer_buckets_raw = cfg.get("easy_prefer_buckets", ["short", "medium"])
    if isinstance(prefer_buckets_raw, str):
        prefer_buckets = {x.strip().lower() for x in prefer_buckets_raw.split(",") if x.strip()}
    elif isinstance(prefer_buckets_raw, list):
        prefer_buckets = {str(x).strip().lower() for x in prefer_buckets_raw if str(x).strip()}
    else:
        prefer_buckets = {"short", "medium"}

    preserve_distribution_default = profile != "easy"
    preserve_label_distribution = bool(cfg.get("preserve_label_distribution", preserve_distribution_default))

    bucket_slack_ratio = float(cfg.get("bucket_slack_ratio", 0.10 if profile == "balanced" else 0.30))
    bucket_slack_ratio = max(0.05, min(0.60, bucket_slack_ratio))

    rng = random.Random(seed)
    ctx = _build_quality_context(records)

    annotated: List[Dict[str, Any]] = []
    primary_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()

    for i, rec in enumerate(records):
        metrics = _quality_components(rec, ctx)
        primary = _primary_label(rec)
        tok_len = len(rec.get("tokens", []) or [])
        ent_count = len(rec.get("entities", []) or [])
        bucket = _length_bucket(tok_len, ctx["p33"], ctx["p66"])
        normalized_text = _normalize_text(str(rec.get("text", "") or ""))

        effective_score = metrics["total"]
        if profile == "easy":
            ease_bonus = 0.10 * metrics["cleanliness"] + 0.08 * metrics["completeness"]
            if bucket in prefer_buckets:
                ease_bonus += 0.06
            if bucket == "long":
                ease_bonus -= 0.05
            if ent_count > 2:
                ease_bonus -= min(0.15, 0.03 * (ent_count - 2))
            effective_score = _safe_float(effective_score + ease_bonus)

        row = {
            "idx": i,
            "score": metrics["total"],
            "effective_score": effective_score,
            "metrics": metrics,
            "primary": primary,
            "bucket": bucket,
            "entity_count": ent_count,
            "token_len": tok_len,
            "norm_text": normalized_text,
            "jitter": rng.random(),
        }
        annotated.append(row)
        primary_counter[primary] += 1
        bucket_counter[bucket] += 1

    eligible = annotated
    easy_filtered_out = 0
    if profile == "easy":
        filtered: List[Dict[str, Any]] = []
        for item in annotated:
            if item["score"] < easy_min_quality:
                easy_filtered_out += 1
                continue
            if easy_max_entities is not None and item["entity_count"] > easy_max_entities:
                easy_filtered_out += 1
                continue
            filtered.append(item)
        if filtered:
            eligible = filtered

    target = min(target_size, len(records))
    primary_quota = _alloc_quota(Counter(x["primary"] for x in eligible), target)
    bucket_quota = _alloc_quota(Counter(x["bucket"] for x in eligible), target)
    bucket_slack = {k: max(1, int(math.ceil(v * bucket_slack_ratio))) for k, v in bucket_quota.items()}

    by_primary: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_primary[item["primary"]].append(item)
        by_bucket[item["bucket"]].append(item)

    def _sort_key(x: Dict[str, Any]) -> Tuple[float, float]:
        return (x["effective_score"], x["jitter"])

    global_sorted = sorted(eligible, key=_sort_key, reverse=True)
    for key in by_primary:
        by_primary[key].sort(key=_sort_key, reverse=True)
    for key in by_bucket:
        by_bucket[key].sort(key=_sort_key, reverse=True)

    selected_items: List[Dict[str, Any]] = []
    selected_idx = set()
    seen_text = set()
    selected_primary_counter: Counter[str] = Counter()
    selected_bucket_counter: Counter[str] = Counter()

    def _pick(item: Dict[str, Any], enforce_unique_text: bool = True) -> bool:
        idx = item["idx"]
        if idx in selected_idx:
            return False
        norm_text = item["norm_text"]
        if enforce_unique_text and norm_text and norm_text in seen_text:
            return False
        selected_items.append(item)
        selected_idx.add(idx)
        if norm_text:
            seen_text.add(norm_text)
        selected_primary_counter[item["primary"]] += 1
        selected_bucket_counter[item["bucket"]] += 1
        return True

    if preserve_label_distribution:
        primary_cursor: Dict[str, int] = {k: 0 for k in by_primary.keys()}
        for label, quota in sorted(primary_quota.items(), key=lambda kv: kv[1], reverse=True):
            rows = by_primary.get(label, [])
            while selected_primary_counter[label] < quota and len(selected_items) < target:
                cur = primary_cursor.get(label, 0)
                if cur >= len(rows):
                    break
                item = rows[cur]
                primary_cursor[label] = cur + 1

                bucket = item["bucket"]
                upper = bucket_quota.get(bucket, 0) + bucket_slack.get(bucket, 1)
                if selected_bucket_counter[bucket] >= max(upper, 1):
                    continue
                _pick(item, enforce_unique_text=True)

        bucket_cursor: Dict[str, int] = {k: 0 for k in by_bucket.keys()}
        for bucket, quota in sorted(bucket_quota.items(), key=lambda kv: kv[1], reverse=True):
            rows = by_bucket.get(bucket, [])
            while selected_bucket_counter[bucket] < quota and len(selected_items) < target:
                cur = bucket_cursor.get(bucket, 0)
                if cur >= len(rows):
                    break
                item = rows[cur]
                bucket_cursor[bucket] = cur + 1
                _pick(item, enforce_unique_text=True)
    else:
        # Keep minimum label diversity while prioritizing easy/high-quality samples.
        for label in sorted(by_primary.keys()):
            if len(selected_items) >= target:
                break
            if label == "__NO_ENTITY__":
                continue
            rows = by_primary.get(label, [])
            for item in rows:
                if _pick(item, enforce_unique_text=True):
                    break

        easy_target = max(len(selected_items), int(round(target * easy_top_fraction)))
        easy_target = min(target, easy_target)
        for item in global_sorted:
            if len(selected_items) >= easy_target:
                break
            _pick(item, enforce_unique_text=True)

    for item in global_sorted:
        if len(selected_items) >= target:
            break
        _pick(item, enforce_unique_text=True)

    fallback_sorted = []
    if len(selected_items) < target and len(eligible) < len(annotated):
        fallback_sorted = sorted(annotated, key=_sort_key, reverse=True)
        for item in fallback_sorted:
            if len(selected_items) >= target:
                break
            _pick(item, enforce_unique_text=True)

    for item in global_sorted:
        if len(selected_items) >= target:
            break
        _pick(item, enforce_unique_text=False)

    if len(selected_items) < target and fallback_sorted:
        for item in fallback_sorted:
            if len(selected_items) >= target:
                break
            _pick(item, enforce_unique_text=False)

    selected_records = [records[x["idx"]] for x in selected_items]

    full_entity_counter = _entity_label_counts(records)
    selected_entity_counter = _entity_label_counts(selected_records)
    full_probs = _to_probs(full_entity_counter)
    selected_probs = _to_probs(selected_entity_counter)

    full_quality_total = [x["metrics"]["total"] for x in annotated]
    selected_quality_total = [x["metrics"]["total"] for x in selected_items]
    selected_effective_total = [x["effective_score"] for x in selected_items]

    metric_keys = ["length", "entity_coverage", "diversity", "semantic", "completeness", "cleanliness", "total"]
    full_metric_means = {k: _mean([x["metrics"][k] for x in annotated]) for k in metric_keys}
    selected_metric_means = {k: _mean([x["metrics"][k] for x in selected_items]) for k in metric_keys}

    report = {
        "source_count": len(records),
        "selected_count": len(selected_records),
        "target_count": target_size,
        "insufficient_source": len(records) < target_size,
        "sampling": {
            "profile": profile,
            "preserve_label_distribution": preserve_label_distribution,
            "easy_top_fraction": easy_top_fraction,
            "easy_min_quality": easy_min_quality,
            "easy_max_entities": easy_max_entities,
            "easy_filtered_out": easy_filtered_out,
            "bucket_slack_ratio": bucket_slack_ratio,
        },
        "quality": {
            "full_mean": _mean(full_quality_total),
            "selected_mean": _mean(selected_quality_total),
            "selected_effective_mean": _mean(selected_effective_total),
            "full_component_means": full_metric_means,
            "selected_component_means": selected_metric_means,
        },
        "distribution": {
            "primary_label_full_counts": dict(primary_counter),
            "primary_label_selected_counts": dict(selected_primary_counter),
            "entity_label_full_counts": dict(full_entity_counter),
            "entity_label_selected_counts": dict(selected_entity_counter),
            "entity_label_full_probs": full_probs,
            "entity_label_selected_probs": selected_probs,
            "entity_label_js_divergence": _js_divergence(full_probs, selected_probs),
        },
        "length": {
            "bucket_full_counts": dict(bucket_counter),
            "bucket_selected_counts": dict(selected_bucket_counter),
            "selected_token_len_mean": _mean([float(x["token_len"]) for x in selected_items]),
            "selected_entity_count_mean": _mean([float(x["entity_count"]) for x in selected_items]),
            "quantiles": {
                "p33": ctx["p33"],
                "p66": ctx["p66"],
                "median": ctx["median_len"],
            },
        },
    }
    return selected_records, report


__all__ = ["sample_records_with_quality_distribution", "hard_filter_records", "is_clean_entity"]
