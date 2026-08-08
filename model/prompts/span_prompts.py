from __future__ import annotations

import json
from typing import Dict, List, Optional

from model.prompts.type_prompts import _LANG_NAMES


def _format_demo(demo: Dict, label_schema: List[str]) -> str:
    """Format one few-shot demo record as a prompt block."""
    text = str(demo.get("text", ""))
    entities = demo.get("entities", []) or []
    schema = label_schema or ["PER", "ORG", "LOC"]
    spans = []
    for i, e in enumerate(entities):
        label = str(e.get("label", schema[0])).upper()
        if label not in schema:
            label = schema[0]
        spans.append({
            "mention": e.get("text", ""),
            "start": e.get("start", 0),
            "end": e.get("end", 1),
            "type_guess": label,
            "confidence": 0.95,
            "reason": "gold",
        })
    return f'Text: {text}\n{json.dumps({"spans": spans}, ensure_ascii=False)}'


def build_span_prompt(
    text: str,
    language: str,
    label_schema: Optional[List[str]] = None,
    demo_examples: Optional[List[Dict]] = None,
) -> str:
    schema = label_schema or ["PER", "ORG", "LOC"]
    type_hint = " / ".join(schema)
    lang_name = _LANG_NAMES.get(str(language).strip().lower(), str(language).upper())

    few_shot_block = ""
    if demo_examples:
        examples_str = "\n\n".join(_format_demo(d, schema) for d in demo_examples)
        few_shot_block = f"\nExamples:\n{examples_str}\n\nNow extract from:\n"

    schema_upper = [x.upper() for x in schema]
    misc_hint = ""
    if "MISC" in schema_upper:
        misc_hint = (
            "3b) MISC extraction: Also extract nationality words and demonyms "
            "(e.g., 'European', 'German', 'Muslim', 'Jewish', 'Thai', 'Spanish', 'Serbian', 'Dutch', 'Hungarian') "
            "— these are valid MISC named entities in CoNLL-style NER even when used as adjectives. "
            "Also extract: disease/medical terms (BSE, encephalopathy), event names (World Cup, Olympics, Grammy), "
            "political/religious group names (Social Democrats, conservatives, Shi'ite), "
            "and acronyms/abbreviations that name entities (GATT, ESB, NATO).\n"
        )
    elif "DATE" in schema_upper:
        misc_hint = (
            "3b) DATE extraction: Also extract temporal expressions that function as named entities "
            "— specific dates, years, named time periods (e.g., 'Monday', 'January 2023', '1990s', "
            "'last Tuesday', 'World War II era', 'Independence Day'). Include these as DATE type_guess spans.\n"
        )

    return (
        "You are SpanAgent in KARVE.\n"
        "Task: Extract ALL potential named entity spans from the text.\n"
        "Focus on WHERE entities are (span boundaries). Give a rough type guess.\n"
        "Rules:\n"
        "1) Span must be a contiguous surface form in the input text.\n"
        "2) Indices are token-based (whitespace split), end is exclusive.\n"
        "3) Be aggressive: include ALL potential named entities — people, places, organizations, products, works, etc.\n"
        f"{misc_hint}"
        "4) Skip standalone punctuation, digits-only, and common function words.\n"
        f"5) type_guess: your best estimate from {type_hint}, or the closest type.\n"
        "6) Output strict JSON only.\n\n"
        f"Language: {lang_name}\n"
        f"{few_shot_block}"
        f"Text: {text}\n\n"
        "Return format:\n"
        '{"spans": [{"mention": "...", "start": 0, "end": 1, "type_guess": "PER", "confidence": 0.9, "reason": "..."}]}\n'
    )
