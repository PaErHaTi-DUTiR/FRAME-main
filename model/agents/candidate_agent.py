from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional

from model.backbones.base_backend import BaseLLMBackend
from model.prompts.candidate_prompts import build_candidate_prompt
from utils.json_parser import parse_json


@dataclass
class CandidateMention:
    mention: str
    start: int
    end: int
    entity_type: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            "mention": self.mention,
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class CandidateAgent:
    """LLM-centered candidate extraction with deterministic fallback."""

    def __init__(self, backend: Optional[BaseLLMBackend] = None, label_schema: Optional[List[str]] = None, config: Optional[Dict] = None):
        self.backend = backend
        self.config = config or {}
        raw_schema = label_schema or ["PER", "ORG", "LOC", "MISC"]
        normalized: List[str] = []
        for x in raw_schema:
            lab = str(x).strip().upper().replace("-", "_")
            if not lab:
                continue
            if lab not in normalized:
                normalized.append(lab)
        self.label_schema = normalized or ["PER", "ORG", "LOC", "MISC"]
        self.label_alias = {
            "PERSON": "PER",
            "PEOPLE": "PER",
            "HUMAN": "PER",
            "GPE": "LOC",
            "FAC": "LOC",
            "LOCATION": "LOC",
            "PLACE": "LOC",
            "COMPANY": "ORG",
            "INSTITUTION": "ORG",
            "WORK_OF_ART": "MISC",
            "EVENT": "MISC",
        }
        # Common discourse/temporal words that are usually not entity mentions.
        self.fallback_stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "that", "this", "these", "those",
            "before", "after", "afterward", "during", "while", "when", "where", "because", "although",
            "however", "therefore", "meanwhile", "shortly", "today", "yesterday", "tomorrow", "monday",
            "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
            "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        }
        self.org_keywords = {
            "inc", "corp", "co", "ltd", "llc", "group", "bank", "agency", "committee", "team", "university", "college",
            "institute", "ministry", "department", "club", "association",
        }
        self.loc_keywords = {
            "city", "state", "province", "county", "district", "street", "river", "lake", "mount", "island", "bay", "republic",
            "kingdom", "capital", "region", "village", "town",
        }

    def extract(self, text: str, language: str, demo_examples: Optional[List[Dict]] = None) -> List[CandidateMention]:
        fallback_candidates = self._extract_with_fallback(text=text)

        if self.backend is not None:
            llm_candidates = self._extract_with_llm(text=text, language=language, demo_examples=demo_examples)
            if llm_candidates:
                return self._fuse_candidates(llm_candidates=llm_candidates, rule_candidates=fallback_candidates)

        return fallback_candidates

    def _extract_with_llm(self, text: str, language: str, demo_examples: Optional[List[Dict]] = None) -> List[CandidateMention]:
        sc_samples = int(self.config.get("sc_samples", 1))
        sc_threshold = float(self.config.get("sc_threshold", 0.5))

        all_mentions = []
        for _ in range(sc_samples):
            prompt = build_candidate_prompt(text=text, language=language, label_schema=self.label_schema, demo_examples=demo_examples)
            result = self.backend.generate(prompt)
            payload = parse_json(result.get("text", ""), default={})
            rows = payload.get("candidates", []) if isinstance(payload, dict) else []
            tokens = text.split()

            mentions: List[CandidateMention] = []
            for row in rows:
                try:
                    mention = str(row.get("mention", "")).strip()
                    start = int(row.get("start", -1))
                    end = int(row.get("end", -1))
                    etype = self._canonical_label(str(row.get("type", self._default_label())))
                    conf = float(row.get("confidence", 0.5))
                    reason = str(row.get("reason", "llm_candidate"))
                    if not mention:
                        continue

                    repaired = self._repair_span(mention=mention, start=start, end=end, tokens=tokens)
                    if repaired is None:
                        continue

                    rs, re = repaired
                    # Trim trailing punctuation-only tokens (e.g. "West Norwood Cemetery .")
                    while re > rs and not self._strip_token(tokens[re - 1]):
                        re -= 1
                    mention_text = " ".join(tokens[rs:re]).strip() if tokens else mention
                    if not mention_text:
                        mention_text = mention

                    if self._is_noise_mention(mention_text):
                        continue

                    mentions.append(
                        CandidateMention(
                            mention=mention_text,
                            start=rs,
                            end=re,
                            entity_type=etype,
                            confidence=max(0.0, min(conf, 1.0)),
                            reason=reason,
                        )
                    )
                except Exception:
                    continue

            deduped = self._dedup(mentions)
            all_mentions.extend(deduped)

        if not all_mentions:
            return []

        # Self-consistency aggregation: majority vote on span + entity_type
        votes = {}
        for m in all_mentions:
            key = (m.start, m.end, m.entity_type)
            if key not in votes:
                votes[key] = {
                    "count": 0,
                    "confidence_sum": 0.0,
                    "mention": m.mention,
                    "reason": m.reason
                }
            votes[key]["count"] += 1
            votes[key]["confidence_sum"] += m.confidence

        final_mentions = []
        for key, vinfo in votes.items():
            vote_ratio = vinfo["count"] / sc_samples
            if vote_ratio >= sc_threshold:
                avg_conf = vinfo["confidence_sum"] / vinfo["count"]
                final_mentions.append(CandidateMention(
                    mention=vinfo["mention"],
                    start=key[0],
                    end=key[1],
                    entity_type=key[2],
                    confidence=avg_conf,
                    reason=f"{vinfo['reason']}|sc_votes:{vinfo['count']}/{sc_samples}"
                ))

        return self._dedup(final_mentions)


    def _extract_with_fallback(self, text: str) -> List[CandidateMention]:
        tokens = text.split()
        mentions: List[CandidateMention] = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if self._is_capitalized_token(tok):
                j = i + 1
                while j < len(tokens) and self._is_capitalized_token(tokens[j]):
                    j += 1
                span_text = " ".join(tokens[i:j])
                if self._is_noise_mention(span_text):
                    i = j
                    continue

                span_tokens = [self._strip_token(x) for x in tokens[i:j] if self._strip_token(x)]
                etype = self._canonical_label(self._fallback_type(span_tokens))
                conf = 0.52 if etype in {"PER", "ORG", "LOC"} else 0.40
                mentions.append(
                    CandidateMention(
                        mention=span_text,
                        start=i,
                        end=j,
                        entity_type=etype,
                        confidence=conf,
                        reason="fallback_capitalized_span_rule",
                    )
                )
                i = j
                continue
            i += 1

        return self._dedup(mentions)

    def _fuse_candidates(self, llm_candidates: List[CandidateMention], rule_candidates: List[CandidateMention]) -> List[CandidateMention]:
        if not llm_candidates:
            return self._dedup(rule_candidates)
        if not rule_candidates:
            return self._dedup(llm_candidates)

        fused: List[CandidateMention] = list(llm_candidates)
        for rc in rule_candidates:
            overlaps = self._overlaps_any(rc, llm_candidates)
            if overlaps and rc.entity_type == "MISC":
                continue
            if overlaps and rc.confidence < 0.52:
                continue

            fused.append(
                CandidateMention(
                    mention=rc.mention,
                    start=rc.start,
                    end=rc.end,
                    entity_type=rc.entity_type,
                    confidence=rc.confidence,
                    reason=f"{rc.reason}|rule_recall_fusion",
                )
            )

        return self._dedup(fused)

    def _overlaps_any(self, cand: CandidateMention, others: List[CandidateMention]) -> bool:
        for o in others:
            if cand.start < o.end and o.start < cand.end:
                return True
        return False

    def _dedup(self, mentions: List[CandidateMention]) -> List[CandidateMention]:
        uniq: Dict[tuple, CandidateMention] = {}
        for m in mentions:
            key = (m.start, m.end)
            if key not in uniq:
                uniq[key] = m
                continue

            cur = uniq[key]
            cur_rank = self._candidate_rank(cur)
            new_rank = self._candidate_rank(m)
            if new_rank > cur_rank:
                uniq[key] = m
        return sorted(uniq.values(), key=lambda x: (x.start, x.end))

    def _candidate_rank(self, mention: CandidateMention) -> float:
        # Prefer high-confidence and non-MISC labels when spans collide.
        type_bonus = 0.04 if mention.entity_type != "MISC" else 0.0
        length_bonus = min(0.05, 0.01 * max(0, len(mention.mention.split()) - 1))
        return mention.confidence + type_bonus + length_bonus

    def _default_label(self) -> str:
        return "MISC" if "MISC" in self.label_schema else self.label_schema[0]

    def _canonical_label(self, label: str) -> str:
        x = str(label).strip().upper().replace("-", "_").replace(" ", "_")
        x = self.label_alias.get(x, x)
        if x in self.label_schema:
            return x
        return self._default_label()

    def _is_capitalized_token(self, token: str) -> bool:
        t = self._strip_token(token)
        if not t:
            return False
        return t[0].isupper() or (t.isupper() and any(ch.isalpha() for ch in t))

    def _strip_token(self, token: str) -> str:
        return re.sub(r"(^[^\w]+|[^\w]+$)", "", str(token))

    def _normalize_token(self, token: str) -> str:
        t = self._strip_token(token)
        return t.lower()

    def _normalize_phrase(self, text: str) -> List[str]:
        parts = [self._normalize_token(x) for x in str(text).split()]
        return [x for x in parts if x]

    def _repair_span(self, mention: str, start: int, end: int, tokens: List[str]) -> Optional[tuple[int, int]]:
        n = len(tokens)
        if n == 0:
            return None

        mention_norm = self._normalize_phrase(mention)
        if not mention_norm:
            return None

        span_len = len(mention_norm)

        def span_match(i: int, j: int) -> bool:
            if i < 0 or j > n or j <= i:
                return False
            return self._normalize_phrase(" ".join(tokens[i:j])) == mention_norm

        if 0 <= start < end <= n and span_match(start, end):
            return start, end

        # Try around hinted start for common off-by-one errors.
        if 0 <= start < n:
            lo = max(0, start - 2)
            hi = min(n - span_len, start + 2)
            for i in range(lo, hi + 1):
                j = i + span_len
                if span_match(i, j):
                    return i, j

        # Global search by normalized mention tokens.
        for i in range(0, n - span_len + 1):
            j = i + span_len
            if span_match(i, j):
                return i, j

        # Single-token fuzzy match fallback.
        if span_len == 1:
            needle = mention_norm[0]
            for i, tok in enumerate(tokens):
                norm = self._normalize_token(tok)
                if norm and (norm == needle or needle in norm or norm in needle):
                    return i, i + 1

        return None

    def _is_noise_mention(self, mention: str) -> bool:
        words = self._normalize_phrase(mention)
        if not words:
            return True
        if all(not any(ch.isalpha() for ch in w) for w in words):
            return True
        if len(words) == 1:
            w = words[0]
            if w in self.fallback_stopwords:
                return True
            if len(w) <= 2 and w.isalpha():
                return True
        if all(w in self.fallback_stopwords for w in words):
            return True
        return False

    def _fallback_type(self, span_tokens: List[str]) -> str:
        if not span_tokens:
            return self._default_label()

        low = [x.lower() for x in span_tokens]
        if any(x in self.org_keywords for x in low):
            return "ORG"
        if any(x in self.loc_keywords for x in low):
            return "LOC"

        if len(span_tokens) >= 2:
            # Multi-token title-case spans are often person mentions.
            if all(x and x[0].isupper() for x in span_tokens):
                return "PER"

        if len(span_tokens) == 1 and span_tokens[0].isupper() and len(span_tokens[0]) <= 5:
            return "ORG"

        return self._default_label()
