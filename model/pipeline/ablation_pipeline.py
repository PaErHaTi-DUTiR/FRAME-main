from __future__ import annotations

from typing import Dict, List, Optional

from model.backbones.base_backend import BaseLLMBackend
from model.pipeline.baseline_pipeline import NaiveRAGNERPipeline, SingleAgentBaselinePipeline
from model.pipeline.karve_x_pipeline import KARVEXPipeline


class AblationPipeline:
    """Unified pipeline switcher for KARVE-X and baselines."""

    def __init__(
        self,
        mode: str,
        backend: Optional[BaseLLMBackend] = None,
        backend_config: Optional[Dict] = None,
        label_schema: Optional[List[str]] = None,
        cache_dir: str = "outputs/cache",
        top_k_evidence: int = 3,
        pipeline_overrides: Optional[Dict] = None,
        calibration_config: Optional[Dict] = None,
    ):
        self.mode = self._normalize_mode(mode)
        overrides = pipeline_overrides or {}
        label_schema = label_schema or (overrides.get("label_schema") if overrides.get("label_schema") else None)

        if self.mode == "karve_x":
            self.pipeline = KARVEXPipeline(
                backend=backend,
                backend_config=backend_config,
                label_schema=label_schema,
                pipeline_config=overrides,
            )
        elif self.mode == "single_agent_baseline":
            self.pipeline = SingleAgentBaselinePipeline(
                backend=backend,
                backend_config=backend_config,
                label_schema=label_schema,
            )
        elif self.mode == "naive_rag_ner_baseline":
            self.pipeline = NaiveRAGNERPipeline(
                backend=backend,
                backend_config=backend_config,
                label_schema=label_schema,
                cache_dir=cache_dir,
                top_k_evidence=int(overrides.get("top_k_evidence", top_k_evidence)),
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported: karve_x, single_agent_baseline, naive_rag_ner_baseline")

    def predict_sample(self, sample: Dict) -> Dict:
        out = self.pipeline.predict_sample(sample)
        out["mode"] = self.mode
        return out

    def predict(self, samples: List[Dict]) -> List[Dict]:
        return [self.predict_sample(s) for s in samples]

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        norm = str(mode).strip().lower().replace("-", "_")
        alias = {
            "karve_x": "karve_x",
            "single_agent": "single_agent_baseline",
            "single_agent_baseline": "single_agent_baseline",
            "naive_rag": "naive_rag_ner_baseline",
            "naive_rag_ner": "naive_rag_ner_baseline",
            "naive_rag_ner_baseline": "naive_rag_ner_baseline",
        }
        return alias.get(norm, norm)


def get_supported_modes() -> List[str]:
    return [
        "karve_x",
        "single_agent_baseline",
        "naive_rag_ner_baseline",
    ]
