from __future__ import annotations

import json
from typing import Dict, List, Optional

# MultiCoNER v2 fine-grained marker labels (detect MC2 schema)
_MC2_MARKERS = frozenset(["OTHERPER", "ARTIST", "ATHLETE", "HUMANSETTLEMENT", "VISUALWORK", "WRITTENWORK"])


def _normalize_label(x: str) -> str:
    return str(x).strip().upper().replace("-", "_").replace("/", "_")


def _is_mc2_schema(normalized: List[str]) -> bool:
    return bool(set(normalized) & _MC2_MARKERS)


def build_candidate_prompt(
    text: str,
    language: str,
    label_schema: List[str],
    demo_examples: Optional[List[Dict]] = None,
) -> str:
    raw_labels = [str(x).strip() for x in (label_schema or []) if str(x).strip()]
    normalized = [_normalize_label(x) for x in raw_labels]
    # Display labels in original case for readability
    labels = ", ".join(raw_labels) if raw_labels else "PER, ORG, LOC, MISC"
    type_options = "|".join(raw_labels) if raw_labels else "PER|ORG|LOC|MISC"

    is_mc2 = _is_mc2_schema(normalized)

    rule_three = (
        "3) Extract ALL potential named entities (high recall). The downstream Verification Agent will filter out bad ones.\n"
        if (not normalized or "MISC" in normalized or is_mc2)
        else "3) Assign only one of the provided labels; do not invent new label types.\n"
    )

    if is_mc2:
        rule_four = (
            "4) FINE-GRAINED TYPE DEFINITIONS (MultiCoNER v2 — 33 types):\n"
            "   Person: OtherPER (generic person), Artist (musician/painter/dancer/actor), "
            "Athlete (sports player), Cleric (religious figure), Politician, "
            "Scientist (researcher/academic/inventor), SportsManager (coach/sports manager)\n"
            "   Organization: ORG (generic org), PublicCorp (state-owned corporation), "
            "PrivateCorp (privately-held company/corporation), "
            "SportsGRP (sports team/group), MusicalGRP (band/orchestra), "
            "CarManufacturer (automobile company), AerospaceManufacturer (aircraft/spacecraft company)\n"
            "   Location: HumanSettlement (city/town/village/district), "
            "OtherLOC (geographic place, country, region, river, mountain), "
            "Facility (named building/stadium/hospital/airport/prison — a physical structure), "
            "Station (train station/bus station/metro station/named transit stop)\n"
            "   Product: Software (app/OS/platform/game engine), Vehicle (car/plane/ship/train model), "
            "OtherPROD (other named product), Clothing (named garment/fashion brand/clothing item), "
            "Food (named food/cuisine), Drink (named beverage/drink brand)\n"
            "   Medical: Disease (illness/condition/syndrome), Symptom, "
            "AnatomicalStructure (body part/organ), MedicalProcedure (surgery/treatment/diagnostic procedure), "
            "Medication/Vaccine (drug/vaccine/medicine name)\n"
            "   Creative works: ArtWork (painting/sculpture/artwork), MusicalWork (song/album/composition), "
            "WrittenWork (book/article/poem/newspaper/magazine), VisualWork (film/TV show/video game/animation)\n"
            "   When unsure between person subtypes → OtherPER. Between org subtypes → ORG. "
            "PublicCorp vs PrivateCorp: use PrivateCorp for privately owned companies. "
            "Station: only for named transit stops, not generic stations. "
            "Between location subtypes → HumanSettlement for named populated places.\n"
        )
    else:
        rule_four = (
            "4) STRICT BOUNDARY: LOC = natural geographic places only (cities, countries, regions, rivers, mountains). "
            "ORG = everything else that is a named entity but NOT a person and NOT a pure geographic place: "
            "companies, government bodies, political parties, military units, sports teams, "
            "buildings/facilities (hospitals, stadiums, stations, cathedrals, cemeteries, casinos), "
            "creative works (songs, albums, films, TV shows, books), documents/lists, events (elections, sports events), "
            "medical conditions, and any other notable named thing. "
            "PER = human beings only. When unsure between LOC and ORG, prefer ORG. "
            "SPAN RULE: in 'City , State' or 'City , County , State' address patterns, extract the ENTIRE phrase as ONE LOC span, not individual pieces. "
            "Do NOT tag geographic adjectives (e.g. 'Arctic', 'European', 'Antarctic') as LOC — only geographic place nouns. "
            "Latin biological species names (binomial nomenclature, e.g. '' Papilio polyxenes '' or '' Mycalesis perseus '') = tag as LOC.\n"
        )

    demo_block = ""
    if demo_examples:
        lines = ["Demo examples (Strict standard, please mimic exactly):"]
        for idx, demo in enumerate(demo_examples, start=1):
            demo_text = str(demo.get("text", "")).strip()
            if len(demo_text) > 180:
                demo_text = f"{demo_text[:177]}..."

            entities = demo.get("entities", []) if isinstance(demo.get("entities", []), list) else []
            cleaned: List[Dict] = []
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                mention = str(ent.get("text", ent.get("mention", ""))).strip()
                if not mention:
                    continue
                cleaned.append(
                    {
                        "mention": mention,
                        "start": ent.get("start", 0),
                        "end": ent.get("end", 0),
                        "type": str(ent.get("type", ent.get("label", ""))).strip(),
                    }
                )
                if len(cleaned) >= 6:
                    break

            lines.append(f"Example {idx} input: {demo_text}")
            if cleaned:
                lines.append(f"Example {idx} entities:")
                lines.append(json.dumps(cleaned, ensure_ascii=False))
            else:
                lines.append(f"Example {idx} entities: []")
            lines.append("")

        demo_block = "\n".join(lines).strip() + "\n\n"

    return (
        "You are Candidate Agent in KARVE.\n"
        "Goal: High-precision entity extraction for multilingual NER.\n"
        "Rules:\n"
        "1) Mention must be a contiguous surface span appearing in the input text.\n"
        "2) Span offsets are over whitespace-tokenized text, with end as exclusive index.\n"
        f"{rule_three}"
        f"{rule_four}"
        "5) Skip pure function words, standalone punctuation/numbers, and temporal (date/time) discourse words.\n"
        "6) Be Aggressive: If a span looks like a named entity, EXTRACT IT. Do not miss any valid entities (High Recall).\n"
        "7) Output strict JSON only.\n\n"
        f"{demo_block}"
        f"Language: {language}\n"
        f"Label schema: {labels}\n"
        f"Text: {text}\n\n"
        "Return format:\n"
        "{\n"
        "  \"candidates\": [\n"
        f"    {{\"mention\": \"...\", \"start\": 0, \"end\": 1, \"type\": \"{type_options}\", \"confidence\": 0.0, \"reason\": \"...\"}}\n"
        "  ]\n"
        "}\n"
    )
