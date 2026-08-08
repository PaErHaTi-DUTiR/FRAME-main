from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from model.agents.span_agent import SpanAgent, SpanMention
from model.agents.type_agent import TypeAgent
from model.backbones import BaseLLMBackend, create_backend_from_config
from model.prompts.type_prompts import _NON_LATIN

# Languages that use character-level space-separated tokenization in WikiANN
_CHARSEQ_LANGS = {"th", "my"}  # Thai and Burmese


class KARVEXPipeline:
    """KARVE-X: 2 LLM calls per sample.

    Agent 1 (SpanAgent): original-language text → span boundaries + rough type guess.
    Agent 2 (TypeAgent): English translation + candidate spans → precise type labels.
    Rule-based merge: TypeAgent type if confidence >= threshold, else SpanAgent guess.
    Gazetteer: rule-based recall boost from train-set entity dictionary.
    """

    _TYPE_AGENT_THRESHOLD = 0.45
    _SPAN_MIN_CONFIDENCE = 0.55  # pre-filter: drop low-confidence spans before TypeAgent

    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        backend_config: Optional[Dict] = None,
        label_schema: Optional[List[str]] = None,
        span_min_confidence: float = 0.55,
        pipeline_config: Optional[Dict] = None,
    ):
        resolved_backend = backend or create_backend_from_config(backend_config)
        self.label_schema = label_schema or ["PER", "ORG", "LOC"]
        self._span_min_conf = span_min_confidence
        _pcfg = pipeline_config or {}
        cot_mode = bool(_pcfg.get("mc2_cot", False))
        self._disable_type_agent = bool(_pcfg.get("disable_type_agent", False))
        self._disable_gazetteer = bool(_pcfg.get("disable_gazetteer", False))
        # FRAME-Pro: optional reflection stage (audits final entity list
        # using the English translation as semantic anchor). Adds 1 LLM
        # call per sample. Off by default for backwards-compatibility.
        self._enable_reflection = bool(_pcfg.get("enable_reflection", False))

        self.span_agent = SpanAgent(backend=resolved_backend, label_schema=self.label_schema)
        self.type_agent = TypeAgent(backend=resolved_backend, label_schema=self.label_schema, cot_mode=cot_mode)
        self._backend = resolved_backend

        # Lazy-loaded per language
        self._gaz: Dict[str, str] = {}
        self._gaz_lang: Optional[str] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _is_charseq(tokens: List[str], unicode_range: Tuple[str, str]) -> bool:
        """True if >35% of tokens are single chars in the given Unicode range."""
        if not tokens:
            return False
        lo, hi = unicode_range
        count = sum(1 for t in tokens if len(t) == 1 and lo <= t <= hi)
        return count / len(tokens) > 0.35

    @staticmethod
    def _join_charseq(tokens: List[str], unicode_range: Tuple[str, str]) -> Tuple[str, Dict[int, Tuple[int, int]]]:
        """Merge consecutive single-char tokens in the given Unicode range.

        Returns (display_text, d2o) where d2o maps display-token-index →
        (orig_start, orig_end_exclusive) in the original token list.
        """
        lo, hi = unicode_range
        groups: List[Tuple[str, int, int]] = []
        i = 0
        while i < len(tokens):
            if len(tokens[i]) == 1 and lo <= tokens[i] <= hi:
                j = i
                while j < len(tokens) and len(tokens[j]) == 1 and lo <= tokens[j] <= hi:
                    j += 1
                groups.append(("".join(tokens[i:j]), i, j))
                i = j
            else:
                groups.append((tokens[i], i, i + 1))
                i += 1
        display_text = " ".join(g[0] for g in groups)
        d2o = {di: (g[1], g[2]) for di, g in enumerate(groups)}
        return display_text, d2o

    @staticmethod
    def _remap_spans(spans: List[SpanMention], d2o: Dict[int, Tuple[int, int]], orig_tokens: List[str]) -> List[SpanMention]:
        """Map display-level span indices back to original (char-level) token indices."""
        remapped: List[SpanMention] = []
        for s in spans:
            d_start = s.start
            d_end_last = s.end - 1
            if d_start not in d2o or d_end_last not in d2o:
                continue
            orig_start = d2o[d_start][0]
            orig_end = d2o[d_end_last][1]
            mention = " ".join(orig_tokens[orig_start:orig_end])
            remapped.append(SpanMention(
                mention=mention,
                start=orig_start,
                end=orig_end,
                type_guess=s.type_guess,
                confidence=s.confidence,
                reason=s.reason,
            ))
        return remapped

    # ------------------------------------------------------------------
    def predict_sample(self, sample: Dict) -> Dict:
        text = str(sample.get("text", ""))
        trans_text = str(sample.get("trans_text", "")).strip()
        language = str(sample.get("language", "en"))
        lang_code = language.strip().lower()
        dataset_name = str(sample.get("dataset", "wikiann"))

        # --- Thai / Burmese: char-level tokenization reconstruction ---
        # WikiANN Thai/Burmese data uses space-separated individual characters.
        # Join them back into readable words before passing to SpanAgent,
        # then map predicted span indices back to original char-level positions.
        tokens_orig = text.split()
        _THAI_RANGE = ("฀", "๿")
        _MYANMAR_RANGE = ("က", "႟")
        script_range = (
            _THAI_RANGE if lang_code == "th" else
            _MYANMAR_RANGE if lang_code == "my" else
            None
        )
        charseq_mode = False
        d2o: Optional[Dict[int, Tuple[int, int]]] = None
        span_text = text  # text passed to SpanAgent
        if script_range and self._is_charseq(tokens_orig, script_range):
            span_text, d2o = self._join_charseq(tokens_orig, script_range)
            charseq_mode = True

        # --- Step 1: SpanAgent on original text ---
        demo_examples = sample.get("demo_examples") or []
        all_spans_raw = self.span_agent.extract_spans(text=span_text, language=language, demo_examples=demo_examples or None)

        # Remap span indices if char-seq reconstruction was applied
        if charseq_mode and d2o is not None:
            all_spans = self._remap_spans(all_spans_raw, d2o, tokens_orig)
        else:
            all_spans = all_spans_raw

        # --- Step 1b: Confidence pre-filter ---
        # Keep high-confidence spans; hold low-confidence spans as fallback candidates
        spans = [s for s in all_spans if s.confidence >= self._span_min_conf]
        low_conf_spans = [s for s in all_spans if s.confidence < self._span_min_conf]

        final_entities: List[Dict] = []
        # For English with no translation, use the text itself — TypeAgent works natively.
        if lang_code == "en" and not trans_text:
            trans_text = text
        has_translation = bool(trans_text) and not self._disable_type_agent

        if spans and has_translation:
            # --- Step 2a: Build alignment hints for TypeAgent ---
            trans_tokens = trans_text.split()
            mentions = []
            for i, s in enumerate(spans):
                entry: Dict = {
                    "id": i,
                    "mention": s.mention,
                    "type_guess": s.type_guess,
                }
                # Alignment hint: find best matching window in English translation
                en_hint = self._find_en_hint(
                    span=s, lang_code=lang_code,
                    orig_tokens=text.split(), trans_tokens=trans_tokens,
                )
                if en_hint:
                    entry["en_context"] = en_hint
                mentions.append(entry)

            # --- Step 2b: TypeAgent classifies + verifies ---
            classifications = self.type_agent.classify(
                trans_text=trans_text,
                mentions=mentions,
                language=language,
            )
            type_map = {c.span_id: c for c in classifications}

            for i, span in enumerate(spans):
                cls = type_map.get(i)
                if cls and cls.confidence >= self._TYPE_AGENT_THRESHOLD:
                    if cls.entity_type in ("NONE", "O", ""):
                        continue
                    etype = cls.entity_type
                    conf = cls.confidence
                    reason = f"type_agent:{cls.reason}"
                else:
                    etype = span.type_guess
                    conf = span.confidence * 0.80
                    reason = f"span_agent_fallback:{span.reason}"

                final_entities.append({
                    "start": span.start,
                    "end": span.end,
                    "text": span.mention,
                    "label": etype,
                    "confidence": conf,
                    "reason": reason,
                })
        else:
            for span in spans:
                final_entities.append({
                    "start": span.start,
                    "end": span.end,
                    "text": span.mention,
                    "label": span.type_guess,
                    "confidence": span.confidence,
                    "reason": f"span_only:{span.reason}",
                })

        # --- Step 2c (FRAME-Pro): Reflection — re-audit via translation ---
        # The reflection stage gives the LLM the original sentence, its
        # English translation, and the current entity list, then asks for
        # the final list (drop FP, add omissions). Only runs when both
        # a translation is available AND the flag is enabled.
        if self._enable_reflection and has_translation and trans_text:
            refined = self._reflect_entities(
                text=text, trans_text=trans_text,
                language=language, current_entities=final_entities,
            )
            if refined is not None:
                final_entities = refined

        # --- Step 3: Gazetteer recall boost ---
        if not self._disable_gazetteer:
            if lang_code in _CHARSEQ_LANGS:
                gaz_min_tokens = 4
            else:
                gaz_min_tokens = 2  # applies to all scripts; eliminates single-token spurious hits
            self._ensure_gazetteer(language, dataset_name)
            if self._gaz:
                tokens = text.split()
                existing = {(e["start"], e["end"]) for e in final_entities}
                for i in range(len(tokens)):
                    for j in range(i + gaz_min_tokens, min(i + 7, len(tokens) + 1)):
                        span_text = " ".join(tokens[i:j])
                        lab = self._gaz.get(span_text.lower())
                        if lab and (i, j) not in existing:
                            final_entities.append({
                                "start": i,
                                "end": j,
                                "text": span_text,
                                "label": lab,
                                "confidence": 0.65,
                                "reason": "gazetteer",
                            })
                            existing.add((i, j))

        output = dict(sample)
        output["pred_entities"] = final_entities
        # Persist raw per-span signals so downstream tooling (e.g. the
        # hyper-parameter sensitivity sweep) can re-derive predictions
        # under alternative (theta_span, theta_type) without new LLM calls.
        raw_spans = []
        type_lookup = locals().get("type_map", {}) or {}
        for i, span in enumerate(all_spans):
            entry = {
                "start": span.start,
                "end": span.end,
                "mention": span.mention,
                "span_conf": float(span.confidence),
                "span_guess": span.type_guess,
                "type_label": None,
                "type_conf": None,
            }
            # Match against the TypeAgent classification only for spans that
            # actually entered the TypeAgent stage (post span-conf filter).
            try:
                idx_in_filtered = next(
                    j for j, s in enumerate(spans)
                    if s.start == span.start and s.end == span.end
                )
                cls = type_lookup.get(idx_in_filtered)
                if cls is not None:
                    entry["type_label"] = cls.entity_type
                    entry["type_conf"] = float(cls.confidence)
            except (StopIteration, AttributeError):
                pass
            raw_spans.append(entry)
        output["trace"] = {
            "trace_version": "karve_x",
            "sample_id": str(sample.get("id", "")),
            "span_count_total": len(all_spans),
            "span_count_filtered": len(spans),
            "entity_count": len(final_entities),
            "trans_text_used": has_translation,
            "raw_spans": raw_spans,
        }
        return output

    def predict(self, samples: List[Dict]) -> List[Dict]:
        return [self.predict_sample(s) for s in samples]

    # ------------------------------------------------------------------
    def _reflect_entities(
        self,
        text: str,
        trans_text: str,
        language: str,
        current_entities: List[Dict],
    ) -> Optional[List[Dict]]:
        """FRAME-Pro reflection: one LLM call to re-audit the entity list.

        The LLM is shown the original sentence, its English translation,
        and the current entity predictions. It returns a final list with
        the freedom to drop FPs, keep TPs, and add omissions.

        Returns None if the call/parsing fails (pipeline falls back to
        the unrefined list, preserving robustness).
        """
        from model.prompts.reflection_prompts import build_reflection_prompt
        from utils.json_parser import parse_json

        prompt = build_reflection_prompt(
            text=text,
            trans_text=trans_text,
            language=language,
            current_entities=current_entities,
            label_schema=self.label_schema,
        )
        try:
            result = self._backend.generate(prompt)
        except Exception:
            return None

        payload = parse_json(result.get("text", ""), default={})
        if not isinstance(payload, dict):
            return None
        items = payload.get("entities", [])
        if not isinstance(items, list):
            return None

        tokens = text.split()
        n = len(tokens)
        schema_up = {x.upper() for x in self.label_schema}
        out: List[Dict] = []
        seen = set()
        for row in items:
            if not isinstance(row, dict):
                continue
            mention = str(row.get("mention", "")).strip()
            label = str(row.get("label", "")).strip().upper()
            if not mention or label not in schema_up:
                continue
            # Align mention to token offsets via case-insensitive whole-token match
            mt = mention.lower().split()
            L = len(mt)
            if L == 0 or L > n:
                continue
            found = None
            tt = [t.lower() for t in tokens]
            for i in range(n - L + 1):
                if tt[i:i + L] == mt:
                    found = (i, i + L)
                    break
            if not found:
                # Fall back to single-token punctuation-stripped match
                if L == 1:
                    target = mt[0].rstrip(".,;:!?\"'")
                    for i, tok in enumerate(tt):
                        if tok.rstrip(".,;:!?\"'") == target:
                            found = (i, i + 1)
                            break
            if not found:
                continue
            key = (found[0], found[1], label)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "start":      found[0],
                "end":        found[1],
                "text":       " ".join(tokens[found[0]:found[1]]),
                "label":      label,
                "confidence": 0.85,
                "reason":     "reflection",
            })
        return out

    # ------------------------------------------------------------------
    def _find_en_hint(
        self,
        span,
        lang_code: str,
        orig_tokens: List[str],
        trans_tokens: List[str],
    ) -> Optional[str]:
        """Find the most likely English window corresponding to this span.

        For Latin-script languages: try direct fuzzy substring match in translation.
        For non-Latin scripts: use positional alignment — map span position ratio
        in original to the same ratio in the English translation, return a ±2 word window.
        """
        if not trans_tokens:
            return None

        n_orig = len(orig_tokens)
        n_trans = len(trans_tokens)

        # Latin-script: try to find mention directly in translation
        if lang_code not in _NON_LATIN:
            mention_lower = span.mention.lower()
            trans_lower = " ".join(trans_tokens).lower()
            if mention_lower in trans_lower:
                return span.mention  # direct match, no hint needed

            # Fuzzy: match individual significant tokens
            mention_words = [w for w in mention_lower.split() if len(w) > 2]
            matched = [w for w in mention_words if w in trans_lower]
            if matched:
                return " ".join(matched)

        # Positional alignment for non-Latin (or no direct match found)
        span_mid = (span.start + span.end - 1) / 2.0
        ratio = span_mid / max(n_orig - 1, 1)

        center = int(round(ratio * (n_trans - 1)))
        half_len = max(1, (span.end - span.start))
        lo = max(0, center - half_len)
        hi = min(n_trans, center + half_len + 1)

        window = trans_tokens[lo:hi]
        if window:
            return " ".join(window)
        return None

    # ------------------------------------------------------------------
    def _ensure_gazetteer(self, language: str, dataset_name: str = "wikiann") -> None:
        cache_key = f"{dataset_name}/{language}"
        if self._gaz_lang == cache_key:
            return
        self._gaz = {}
        self._gaz_lang = cache_key

        # Try dataset-specific path first, then fall back to WikiANN
        candidates = [
            f"datasets/processed/{dataset_name}/{language}/train.jsonl",
            f"datasets/processed/wikiann/{language}/train.jsonl",
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                            for ent in d.get("entities", []):
                                span_text = ent.get("text", "")
                                lab = ent.get("label", "")
                                if span_text and len(span_text) > 1 and lab:
                                    self._gaz[span_text.lower()] = lab
                        except Exception:
                            continue
                break
