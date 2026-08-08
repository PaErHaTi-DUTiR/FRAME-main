#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import compute_entity_metrics, load_processed_split
from utils import build_run_config, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction file with merged YAML configs")
    parser.add_argument("--global-config", default="configs/global.yaml", help="Global YAML config path")
    parser.add_argument("--experiment", required=True, help="Experiment YAML path")
    parser.add_argument("--predictions", required=True, help="Path to predictions.json")
    parser.add_argument("--output", default="", help="Optional output metrics path")
    args = parser.parse_args()

    cfg = build_run_config(args.global_config, args.experiment)

    paths_cfg = cfg.get("paths", {})
    processed_root = REPO_ROOT / str(paths_cfg.get("processed_root", "datasets/processed"))

    dataset_cfg = cfg.get("dataset", {})
    splits_cfg = cfg.get("splits", {})
    limits_cfg = cfg.get("limits", {})

    dataset_name = str(dataset_cfg.get("name", "wikiann"))
    language = str(dataset_cfg.get("language", "en"))
    eval_split = str(splits_cfg.get("eval", "test"))

    max_samples = limits_cfg.get("max_samples_per_language")
    low_resource_ratio = float(limits_cfg.get("low_resource_ratio", 1.0))
    seed = int(cfg.get("seed", 42))

    gold_records = load_processed_split(
        processed_root,
        dataset_name,
        language,
        eval_split,
        max_samples=max_samples,
        low_resource_ratio=low_resource_ratio,
        seed=seed,
    )

    pred_records = read_json(args.predictions)
    metrics = compute_entity_metrics(pred_records, gold_records)

    if args.output:
        write_json(args.output, metrics)

    print(metrics["micro"])


if __name__ == "__main__":
    main()
