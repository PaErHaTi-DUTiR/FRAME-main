from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence


def _normalize_label(label: str) -> str:
    return str(label).strip().upper().replace("-", "_").replace(" ", "_")


def _entity_set(entities: Sequence[Dict]) -> set:
    return {(int(e["start"]), int(e["end"]), _normalize_label(e["label"])) for e in entities}


def compute_entity_metrics(pred_records: Sequence[Dict], gold_records: Sequence[Dict]) -> Dict:
    gold_by_id = {str(r["id"]): r for r in gold_records}
    pred_by_id = {str(r["id"]): r for r in pred_records}

    tp = fp = fn = 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for sample_id, gold in gold_by_id.items():
        gold_entities = gold.get("entities", [])
        pred_entities = pred_by_id.get(sample_id, {}).get("pred_entities", pred_by_id.get(sample_id, {}).get("entities", []))

        gset = _entity_set(gold_entities)
        pset = _entity_set(pred_entities)
        inter = gset & pset

        tp += len(inter)
        fp += len(pset - gset)
        fn += len(gset - pset)

        for _, _, label in inter:
            per_type[label]["tp"] += 1
        for _, _, label in (pset - gset):
            per_type[label]["fp"] += 1
        for _, _, label in (gset - pset):
            per_type[label]["fn"] += 1

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

    by_type = {}
    for label, c in sorted(per_type.items()):
        lp = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        lr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        lf = (2 * lp * lr / (lp + lr)) if (lp + lr) else 0.0
        by_type[label] = {
            "precision": lp,
            "recall": lr,
            "f1": lf,
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
        }

    return {
        "micro": {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "per_type": by_type,
        "num_gold_samples": len(gold_by_id),
        "num_pred_samples": len(pred_by_id),
    }
