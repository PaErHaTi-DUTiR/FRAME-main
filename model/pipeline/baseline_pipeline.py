from __future__ import annotations

from typing import Dict, List, Optional

from model.agents.candidate_agent import CandidateAgent
from model.agents.retrieval_agent import RetrievalAgent
from model.backbones import BaseLLMBackend, create_backend_from_config
from utils.cache import SimpleFileCache


def _default_alignment(reason: str, status: str = "no_match") -> Dict:
    return {
        "best_evidence": {},
        "best_evidence_id": "",
        "alignment_score": 0.0,
        "status": status,
        "reason": reason,
        "alignment_status": status,
        "alignment_reason": reason,
        "signal_scores": {
            "alias": 0.0,
            "semantic": 0.0,
            "language_bridge": 0.0,
            "source": 0.0,
        },
    }


def _default_verification(reason: str, support: float = 0.0, used_evidences: Optional[List[Dict]] = None) -> Dict:
    decision = "accept" if support >= 0.6 else "revise"
    return {
        "support_score": max(0.0, min(1.0, support)),
        "type_consistency": 0.5,
        "conflict_score": 0.1,
        "final_decision": decision,
        "decision": decision,
        "decision_reason": reason,
        "used_evidences": used_evidences or [],
        "uncertain": decision != "accept",
    }


def _default_reflection(reason: str, confidence: float, trigger_source: str, triggered: bool = False) -> Dict:
    return {
        "keep_entity": True,
        "revised_type": "MISC",
        "confidence": confidence,
        "reason": reason,
        "triggered": triggered,
        "revision_action": "keep",
        "revision_reason": reason,
        "revision_confidence": confidence,
        "trigger_source": trigger_source,
    }


class SingleAgentBaselinePipeline:
    """Single-agent baseline using only candidate extraction output."""

    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        backend_config: Optional[Dict] = None,
        label_schema: Optional[List[str]] = None,
    ):
        resolved_backend = backend or create_backend_from_config(backend_config)
        self.candidate_agent = CandidateAgent(backend=resolved_backend, label_schema=label_schema)

    def predict_sample(self, sample: Dict) -> Dict:
        text = str(sample.get("text", ""))
        language = str(sample.get("language", "en"))
        sample_id = str(sample.get("id", ""))

        candidates = self.candidate_agent.extract(text=text, language=language)
        final_entities: List[Dict] = []
        trace_steps: List[Dict] = []

        for idx, cand in enumerate(candidates):
            final_entity = {
                "start": cand.start,
                "end": cand.end,
                "text": cand.mention,
                "label": cand.entity_type,
                "confidence": cand.confidence,
                "reason": "single_agent_candidate_output",
            }
            final_entities.append(final_entity)

            alignment = _default_alignment(reason="single_agent_no_alignment", status="skipped")
            verification = _default_verification(
                reason="single_agent_no_verification",
                support=cand.confidence,
                used_evidences=[],
            )
            reflection = {
                **_default_reflection(
                    reason="single_agent_no_reflection",
                    confidence=cand.confidence,
                    trigger_source="single_agent",
                    triggered=False,
                ),
                "revised_type": cand.entity_type,
            }

            trace_steps.append(
                {
                    "candidate_id": f"cand_{idx}",
                    "candidate": cand.to_dict(),
                    "retrieval": {
                        "query": cand.mention,
                        "aliases": [cand.mention],
                        "rewrite_reason": "single_agent_no_retrieval",
                        "evidences": [],
                    },
                    "alignment": alignment,
                    "verification": verification,
                    "reflection": reflection,
                    "final_entity": final_entity,
                }
            )

        trace = {
            "trace_version": "karve_v1",
            "sample_id": sample_id,
            "language": language,
            "input_text": text,
            "candidates": [c.to_dict() for c in candidates],
            "steps": trace_steps,
            "final_output": final_entities,
        }

        output = dict(sample)
        output["pred_entities"] = final_entities
        output["trace"] = trace
        output["traces"] = trace_steps
        return output

    def predict(self, samples: List[Dict]) -> List[Dict]:
        return [self.predict_sample(s) for s in samples]


class NaiveRAGNERPipeline:
    """Naive RAG-NER baseline: candidate + retrieval + top-evidence type heuristic."""

    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        backend_config: Optional[Dict] = None,
        label_schema: Optional[List[str]] = None,
        cache_dir: str = "outputs/cache",
        top_k_evidence: int = 3,
    ):
        resolved_backend = backend or create_backend_from_config(backend_config)
        cache = SimpleFileCache(cache_dir)
        self.candidate_agent = CandidateAgent(backend=resolved_backend, label_schema=label_schema)
        self.retrieval_agent = RetrievalAgent(backend=resolved_backend, cache=cache)
        self.top_k_evidence = top_k_evidence

    def predict_sample(self, sample: Dict) -> Dict:
        text = str(sample.get("text", ""))
        language = str(sample.get("language", "en"))
        sample_id = str(sample.get("id", ""))

        candidates = self.candidate_agent.extract(text=text, language=language)
        final_entities: List[Dict] = []
        trace_steps: List[Dict] = []

        for idx, cand in enumerate(candidates):
            retrieval_bundle = self.retrieval_agent.retrieve_with_trace(
                candidate=cand,
                context=text,
                language=language,
                top_k=self.top_k_evidence,
            )
            evidences = retrieval_bundle.evidences
            best = evidences[0] if evidences else None

            inferred_type = self._infer_type(best.summary if best else "", fallback=cand.entity_type)
            confidence = cand.confidence
            if best is not None:
                confidence = max(0.0, min(1.0, 0.5 * cand.confidence + 0.5 * best.retrieval_score))

            final_entity = {
                "start": cand.start,
                "end": cand.end,
                "text": cand.mention,
                "label": inferred_type,
                "confidence": confidence,
                "reason": "naive_rag_top_evidence",
            }
            final_entities.append(final_entity)

            alignment_score = float(best.retrieval_score) if best is not None else 0.0
            alignment_status = "aligned" if best is not None else "no_match"
            alignment_reason = "naive_rag_top_retrieval_score"
            alignment = {
                "best_evidence": best.to_dict() if best is not None else {},
                "best_evidence_id": best.evidence_id if best is not None else "",
                "alignment_score": alignment_score,
                "status": alignment_status,
                "reason": alignment_reason,
                "alignment_status": alignment_status,
                "alignment_reason": alignment_reason,
                "signal_scores": {
                    "alias": 0.0,
                    "semantic": 0.0,
                    "language_bridge": 0.0,
                    "source": alignment_score,
                },
            }

            decision = "accept" if alignment_score >= 0.45 else "revise"
            verification = {
                "support_score": alignment_score,
                "type_consistency": 0.5,
                "conflict_score": 0.25 if len(evidences) > 1 else 0.1,
                "final_decision": decision,
                "decision": decision,
                "decision_reason": "naive_rag_support_from_retrieval_score",
                "used_evidences": [e.to_dict() for e in evidences[:2]],
                "uncertain": decision != "accept",
            }

            reflection = {
                "keep_entity": True,
                "revised_type": inferred_type,
                "confidence": confidence,
                "reason": "naive_rag_no_constrained_reflection",
                "triggered": False,
                "revision_action": "keep",
                "revision_reason": "naive_rag_no_constrained_reflection",
                "revision_confidence": confidence,
                "trigger_source": "naive_rag",
            }

            trace_steps.append(
                {
                    "candidate_id": f"cand_{idx}",
                    "candidate": cand.to_dict(),
                    "retrieval": {
                        "query": retrieval_bundle.query,
                        "aliases": retrieval_bundle.aliases,
                        "rewrite_reason": retrieval_bundle.rewrite_reason,
                        "evidences": [e.to_dict() for e in evidences],
                    },
                    "alignment": alignment,
                    "verification": verification,
                    "reflection": reflection,
                    "final_entity": final_entity,
                }
            )

        trace = {
            "trace_version": "karve_v1",
            "sample_id": sample_id,
            "language": language,
            "input_text": text,
            "candidates": [c.to_dict() for c in candidates],
            "steps": trace_steps,
            "final_output": final_entities,
        }

        output = dict(sample)
        output["pred_entities"] = final_entities
        output["trace"] = trace
        output["traces"] = trace_steps
        return output

    def predict(self, samples: List[Dict]) -> List[Dict]:
        return [self.predict_sample(s) for s in samples]

    def _infer_type(self, text: str, fallback: str) -> str:
        lower = text.lower()
        keyword_map = {
            "PER": ["born", "actor", "singer", "writer", "person", "politician"],
            "ORG": ["company", "organization", "university", "agency", "club", "corporation"],
            "LOC": ["city", "country", "river", "province", "capital", "located"],
            "MISC": ["event", "language", "product", "festival", "film"],
        }

        best_label = fallback
        best_hit = 0
        for label, keys in keyword_map.items():
            hit = sum(1 for k in keys if k in lower)
            if hit > best_hit:
                best_hit = hit
                best_label = label

        return best_label
