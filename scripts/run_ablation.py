#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import AblationPipeline, compute_entity_metrics, load_processed_split
from utils import build_run_config, ensure_dir, write_json
from utils.memory_retriever import DemoMemoryRetriever


def _run_variant(
    *,
    cfg: Dict,
    run_name: str,
    mode: str,
    pipeline_cfg: Dict,
    backend_cfg: Dict,
    calibration_cfg: Dict,
) -> Dict:
    seed = int(cfg.get("seed", 42))
    paths_cfg = cfg.get("paths", {})
    processed_root = REPO_ROOT / str(paths_cfg.get("processed_root", "datasets/processed"))
    outputs_root = REPO_ROOT / str(paths_cfg.get("outputs_root", "outputs"))

    dataset_cfg = cfg.get("dataset", {})
    splits_cfg = cfg.get("splits", {})
    limits_cfg = cfg.get("limits", {})

    dataset_name = str(dataset_cfg.get("name", "wikiann"))
    language = str(dataset_cfg.get("language", "en"))
    eval_split = str(splits_cfg.get("eval", "test"))

    max_samples = limits_cfg.get("max_samples_per_language")
    low_resource_ratio = float(limits_cfg.get("low_resource_ratio", 1.0))
    quality_sample = bool(limits_cfg.get("quality_sample", False))

    eval_records = load_processed_split(
        processed_root,
        dataset_name,
        language,
        eval_split,
        max_samples=max_samples,
        low_resource_ratio=low_resource_ratio,
        seed=seed,
        quality_sample=quality_sample,
    )
    for rec in eval_records:
        if not rec.get("dataset"):
            rec["dataset"] = dataset_name

    candidate_cfg = pipeline_cfg.get("candidate", {})
    demo_k = candidate_cfg.get("demo_k", 0)
    if demo_k > 0:
        # Load train records as memory pool
        train_records = load_processed_split(
            processed_root,
            dataset_name,
            language,
            "train",
            max_samples=min(2000, max_samples * 5 if max_samples else 2000),
            low_resource_ratio=low_resource_ratio,
            seed=seed,
        )
        retriever = DemoMemoryRetriever(
            demo_records=train_records,
            k=demo_k,
            mode=candidate_cfg.get("demo_mode", "dense"),
            label_schema=pipeline_cfg.get("label_schema"),
        )
        for rec in eval_records:
            rec["demo_examples"] = retriever.retrieve(rec["text"])

    pipeline = AblationPipeline(
        mode=mode,
        backend_config=backend_cfg,
        pipeline_overrides=pipeline_cfg,
        calibration_config=calibration_cfg,
    )

    predictions: List[Dict] = []
    for i, rec in enumerate(tqdm(eval_records, desc=f"ablation:{run_name}")):
        try:
            predictions.append(pipeline.predict_sample(rec))
        except RuntimeError as exc:
            # Content safety filter or transient API error on a single sample — skip it.
            import logging as _logging
            _logging.getLogger("karve.runner").warning(
                "Skipping sample id=%s due to error: %s", rec.get("id", i), exc
            )
            skipped = dict(rec)
            skipped["pred_entities"] = []
            skipped["trace"] = {"error": str(exc), "skipped": True}
            predictions.append(skipped)
        if (i + 1) % 50 == 0:
            try:
                tmp_dir = outputs_root / "runs" / f"{run_name}_partial"
                ensure_dir(tmp_dir)
                write_json(tmp_dir / "predictions_partial.json", predictions)
            except Exception: pass

    metrics = compute_entity_metrics(predictions, eval_records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(outputs_root / "runs" / f"{run_name}_{timestamp}")
    ensure_dir(outputs_root / "metrics")

    write_json(run_dir / "predictions.json", predictions)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "config_snapshot.json", cfg)

    return {
        "run_name": run_name,
        "mode": mode,
        "backend_mode": str(backend_cfg.get("mode", "off")),
        "run_dir": str(run_dir),
        "micro": metrics["micro"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KARVE ablation suite")
    parser.add_argument("--global-config", default="configs/global.yaml", help="Global YAML config path")
    parser.add_argument("--experiment", default="configs/experiments/ablation_modes.yaml", help="Ablation YAML config path")
    args = parser.parse_args()

    cfg = build_run_config(args.global_config, args.experiment)
    variants = cfg.get("ablation_variants", [])
    if not variants:
        raise ValueError("ablation config requires ablation_variants")

    base_run_name = str(cfg.get("run_name", "karve_ablation_suite"))
    base_pipeline_cfg = dict(cfg.get("pipeline", {}))
    base_backend_cfg = dict(cfg.get("backend", {}))
    base_calibration_cfg = dict(cfg.get("calibration", {})) if isinstance(cfg.get("calibration", {}), dict) else {}

    summary = []
    for variant in variants:
        variant_name = str(variant.get("name", "variant"))
        variant_mode = str(variant.get("mode", variant_name))
        variant_pipeline_cfg = dict(base_pipeline_cfg)
        variant_pipeline_cfg.update(variant.get("pipeline", {}))

        variant_backend_cfg = dict(base_backend_cfg)
        variant_backend_cfg.update(variant.get("backend", {}))

        variant_calibration_cfg = dict(base_calibration_cfg)
        if isinstance(variant.get("calibration", {}), dict):
            variant_calibration_cfg.update(variant.get("calibration", {}))

        run_info = _run_variant(
            cfg=cfg,
            run_name=f"{base_run_name}_{variant_name}",
            mode=variant_mode,
            pipeline_cfg=variant_pipeline_cfg,
            backend_cfg=variant_backend_cfg,
            calibration_cfg=variant_calibration_cfg,
        )
        summary.append(run_info)
        print(json.dumps(run_info, ensure_ascii=False))

    summary_path = REPO_ROOT / str(cfg.get("paths", {}).get("outputs_root", "outputs")) / "metrics" / f"{base_run_name}_summary.json"
    write_json(summary_path, {"experiment_type": "ablation_suite", "runs": summary})
    print(f"Saved ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
