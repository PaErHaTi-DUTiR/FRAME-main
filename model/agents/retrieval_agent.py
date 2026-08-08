from __future__ import annotations

from dataclasses import dataclass
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional

from model.agents.candidate_agent import CandidateMention
from model.backbones.base_backend import BaseLLMBackend
from utils.cache import SimpleFileCache


@dataclass
class EvidenceItem:
    evidence_id: str
    title: str
    aliases: List[str]
    summary: str
    language: str
    retrieval_score: float
    source: str
    retrieval_query: str = ""
    retrieval_rank: int = 0
    source_metadata: Optional[Dict] = None

    @property
    def score(self) -> float:
        return self.retrieval_score

    @classmethod
    def from_dict(cls, payload: Dict) -> "EvidenceItem":
        return cls(
            evidence_id=str(payload.get("evidence_id", "")) or "evidence_unknown",
            title=str(payload.get("title", "")),
            aliases=[str(x) for x in payload.get("aliases", [])],
            summary=str(payload.get("summary", "")),
            language=str(payload.get("language", "en")),
            retrieval_score=float(payload.get("retrieval_score", payload.get("score", 0.0))),
            source=str(payload.get("source", "unknown")),
            retrieval_query=str(payload.get("retrieval_query", "")),
            retrieval_rank=int(payload.get("retrieval_rank", 0)),
            source_metadata=payload.get("source_metadata", {}),
        )

    def to_dict(self) -> Dict:
        return {
            "evidence_id": self.evidence_id,
            "title": self.title,
            "aliases": self.aliases,
            "summary": self.summary,
            "language": self.language,
            "retrieval_score": self.retrieval_score,
            "score": self.retrieval_score,
            "source": self.source,
            "retrieval_query": self.retrieval_query,
            "retrieval_rank": self.retrieval_rank,
            "source_metadata": self.source_metadata or {},
        }


@dataclass
class RetrievalBundle:
    query: str
    aliases: List[str]
    rewrite_reason: str
    evidences: List[EvidenceItem]


class RetrievalAgent:
    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        cache: Optional[SimpleFileCache] = None,
        timeout: int = 8,
        offline_knowledge_path: str = "",
        retrieval_mode: str = "offline",
        prefer_offline: bool = True,
    ):
        self.backend = backend
        self.cache = cache
        self.timeout = timeout
        self.requested_mode = str(retrieval_mode).strip().lower() or "offline"
        self.retrieval_mode = "offline"
        self.prefer_offline = True

        self._local_kb_entries: List[Dict] = []
        self._local_kb_index: Dict[str, List[int]] = {}
        self._load_local_knowledge(offline_knowledge_path)

    def _normalize_key(self, key: str) -> str:
        return str(key or "").strip().lower()

    def _load_local_knowledge(self, offline_knowledge_path: str) -> None:
        path = str(offline_knowledge_path or "").strip()
        if not path:
            return
        p = Path(path)
        if not p.exists() or not p.is_file():
            return

        rows: List[Dict] = []
        try:
            if p.suffix.lower() == ".jsonl":
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if isinstance(item, dict):
                            rows.append(item)
            elif p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        rows.extend(data)
            
            self._local_kb_entries = rows
            for idx, r in enumerate(rows):
                aliases = [r.get("title", ""), r.get("id", "")] + r.get("aliases", [])
                for a in set(aliases):
                    k = self._normalize_key(str(a))
                    if k:
                        if k not in self._local_kb_index:
                            self._local_kb_index[k] = []
                        if len(self._local_kb_index[k]) < 50:
                            self._local_kb_index[k].append(idx)
        except:
            pass

    def retrieve(self, candidate: CandidateMention, context: str, language: str, top_k: int = 3) -> List[EvidenceItem]:
        return self.retrieve_with_trace(candidate, context, language, top_k).evidences

    def retrieve_with_trace(
        self, candidate: CandidateMention, context: str, language: str, top_k: int = 3
    ) -> RetrievalBundle:
        query = candidate.mention.strip()
        aliases = [query, query.lower()]
        rewrite_reason = "fallback"

        alias_sig = "|".join(sorted({self._normalize_key(a) for a in aliases if self._normalize_key(a)}))
        cache_key = f"retrieval_v7::{language}::{query}::{alias_sig}::{top_k}"
        
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("evidences"), list):
                evidences = [EvidenceItem.from_dict(x) for x in cached.get("evidences", [])][:top_k]
                return RetrievalBundle(
                    query=str(cached.get("query", query)),
                    aliases=[str(x) for x in cached.get("aliases", aliases)],
                    rewrite_reason=str(cached.get("rewrite_reason", rewrite_reason)),
                    evidences=evidences,
                )

        evidences = self._offline_retrieve(candidate, query, aliases, context, language, top_k)
        if not evidences:
            evidences = self._fallback_evidence(candidate, context, language)

        bundle = RetrievalBundle(query=query, aliases=aliases, rewrite_reason=rewrite_reason, evidences=evidences[:top_k])
        
        if self.cache is not None:
            self.cache.set(
                cache_key,
                {
                    "query": query,
                    "aliases": aliases,
                    "rewrite_reason": rewrite_reason,
                    "evidences": [e.to_dict() for e in evidences],
                },
                ttl_seconds=3600,
            )
            
        return bundle

    def _offline_retrieve(self, candidate, query, aliases, context, language, top_k):
        evidences = []
        found_indices = set()
        for alias in aliases:
            k = self._normalize_key(alias)
            if k in self._local_kb_index:
                for idx in self._local_kb_index[k]:
                    if idx not in found_indices:
                        found_indices.add(idx)
                        r = self._local_kb_entries[idx]
                        title = r.get("title", k)
                        ev = EvidenceItem(
                            evidence_id=str(r.get("id", f"offline_{idx}")),
                            title=title,
                            aliases=r.get("aliases", []),
                            summary=r.get("description", r.get("summary", "")),
                            language=r.get("language", language),
                            retrieval_score=1.0,
                            source="offline_kb",
                            retrieval_query=query,
                            retrieval_rank=len(evidences) + 1,
                            source_metadata={"provider": "local"}
                        )
                        evidences.append(ev)
                        if len(evidences) >= top_k:
                            return evidences
        return evidences

    def _fallback_evidence(self, candidate, context, language):
        knowledge_dir = "outputs/knowledge"
        os.makedirs(knowledge_dir, exist_ok=True)
        safe_name = "".join(c for c in candidate.mention if c.isalnum() or c in (' ', '_')).replace(' ', '_')
        if not safe_name: safe_name = "UNKNOWN"
        wiki_file = os.path.join(knowledge_dir, f"{language}_{safe_name}.json")
        
        if os.path.exists(wiki_file):
            try:
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                    return [EvidenceItem.from_dict(entry)]
            except:
                pass
                
        try:
            backend = getattr(self, "backend", None)
            if not backend:  # dynamic fallback to global backend via config
                import importlib
                from utils.config_loader import ConfigLoader
                from model.backbones import get_backend
                cfg = ConfigLoader().config
                backend = get_backend(cfg)
                
            prompt = f"Provide a precise, 3-sentence factual Wikipedia-style definition for the entity '{candidate.mention}'. Context where it appeared: '{context}'. Clearly state if it is a Person, Organization, Location, or Product. If it does not exist or you are completely unsure, reply exactly with 'UNKNOWN'."
            sys_prompt = "You are a highly factual Encyclopedia engine. Do not invent details."
            rsp = backend.generate(prompt=prompt, system_prompt=sys_prompt).get("text", "")
            
            if rsp and "UNKNOWN" not in rsp.upper()[:30] and len(rsp) > 10:
                evidence_id = f"llm_wiki_{language}_{safe_name}".lower()
                item = EvidenceItem(
                    evidence_id=evidence_id,
                    title=candidate.mention,
                    aliases=[candidate.mention, candidate.mention.lower()],
                    summary=rsp.strip(),
                    language=language,
                    retrieval_score=0.95,
                    source="llm_wiki",
                    retrieval_query=candidate.mention,
                    retrieval_rank=0,
                    source_metadata={"provider": "llm_generated_wiki"}
                )
                with open(wiki_file, 'w', encoding='utf-8') as wf:
                    json.dump(item.to_dict(), wf, ensure_ascii=False)
                return [item]
        except Exception as e:
            print(f"LLM WIKI FAILURE: {e}")
        return []
