from __future__ import annotations

import json
from typing import Dict, List

# ISO 639-1 code → full name
_LANG_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh", "ko": "Korean",
    "lt": "Lithuanian", "lv": "Latvian", "mr": "Marathi", "ms": "Malay",
    "my": "Burmese", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian",
    "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Filipino",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "yo": "Yoruba", "zh": "Chinese",
    # MC2 language codes
    "zh-tw": "Chinese (Traditional)", "de": "German", "es": "Spanish",
    "fa": "Persian", "ko": "Korean", "nl": "Dutch",
    # MasakhaNER 2.0 ISO 639-3 codes (African low-resource languages)
    "bam": "Bambara", "bbj": "Ghomala", "ewe": "Ewe", "fon": "Fon",
    "hau": "Hausa", "ibo": "Igbo", "kin": "Kinyarwanda", "lug": "Luganda",
    "luo": "Luo", "mos": "Mooré", "nya": "Chichewa", "pcm": "Nigerian Pidgin",
    "sna": "Shona", "swa": "Swahili", "tsn": "Tswana", "twi": "Twi",
    "wol": "Wolof", "xho": "Xhosa", "yor": "Yoruba", "zul": "Zulu",
}

# Languages that use non-Latin scripts — TypeAgent should rely more on English translation
_NON_LATIN = {
    "ar", "bn", "el", "fa", "he", "hi", "hy", "ja", "ka", "kk",
    "ko", "mr", "my", "ru", "sr", "ta", "te", "th", "uk", "ur",
    "uk", "zh", "zh-tw",
}

# Low-resource or noisy-translation languages — warn TypeAgent translation may be imperfect
_LOW_RESOURCE = {
    "yo", "jv", "my", "sw", "tl", "mr", "eu", "lv", "lt", "et",
    # MasakhaNER 2.0 African languages (all low-resource)
    "bam", "bbj", "ewe", "fon", "hau", "ibo", "kin", "lug", "luo",
    "mos", "nya", "pcm", "sna", "swa", "tsn", "twi", "wol", "xho", "yor", "zul",
}

# MC2 fine-grained 33-type schema
_MC2_TYPES = {
    "AEROSPACEMANUFACTURER", "ANATOMICALSTRUCTURE", "ARTWORK", "ARTIST", "ATHLETE",
    "CARMANUFACTURER", "CLERIC", "CLOTHING", "DISEASE", "DRINK", "FACILITY", "FOOD",
    "HUMANSETTLEMENT", "MEDICALPROCEDURE", "MEDICATION/VACCINE", "MUSICALGRP",
    "MUSICALWORK", "ORG", "OTHERLOC", "OTHERPER", "OTHERPROD", "POLITICIAN",
    "PRIVATECORP", "PUBLICCORP", "SCIENTIST", "SOFTWARE", "SPORTSGRP", "SPORTSMANAGER",
    "STATION", "SYMPTOM", "VEHICLE", "VISUALWORK", "WRITTENWORK",
}


def _is_multiconer_schema(schema_upper: list) -> bool:
    # MC2 has 33 types; require >=10 matching types to avoid false positives
    # (CoNLL03 has {PER,ORG,LOC,MISC} and WikiANN has {PER,ORG,LOC}, both contain ORG which is also in MC2)
    return len(set(schema_upper) & _MC2_TYPES) >= 10


def _build_multiconer_type_rules_v2() -> str:
    """Enhanced rules with explicit disambiguation guide and CoT instruction."""
    return (
        "Type definitions for MultiCoNER v2 (33 fine-grained types):\n"
        "\nPERSON types:\n"
        "- OtherPER: persons not fitting specific categories (writers, philosophers, historical figures, unnamed individuals)\n"
        "- Artist: visual artists, painters, sculptors, illustrators, photographers, directors, choreographers\n"
        "- Athlete: sports players, olympians, competitors in any sport\n"
        "- Cleric: religious leaders, priests, popes, imams, monks, rabbis, pastors\n"
        "- Politician: government officials, presidents, prime ministers, senators, mayors, political leaders\n"
        "- Scientist: researchers, academics, engineers, inventors, mathematicians, doctors (named scientists)\n"
        "- SportsManager: sports coaches, team managers, athletic directors, sports executives\n"
        "\nORGANIZATION types:\n"
        "- ORG: general organizations, NGOs, government bodies, universities, schools, hospitals (not fitting below)\n"
        "- AerospaceManufacturer: aerospace/aviation companies and agencies (Boeing, Airbus, SpaceX, NASA programs)\n"
        "- CarManufacturer: automotive companies (Toyota, Ford, Tesla, BMW, Honda)\n"
        "- MusicalGRP: bands, orchestras, choirs, music groups (Beatles, Vienna Philharmonic)\n"
        "- PrivateCorp: privately held companies (not aerospace/auto/musical)\n"
        "- PublicCorp: publicly traded corporations on stock markets\n"
        "- SportsGRP: sports teams, leagues, federations, tournaments as entities (NFL, Premier League, Manchester United)\n"
        "\nLOCATION types:\n"
        "- HumanSettlement: cities, towns, villages, boroughs, districts, neighborhoods\n"
        "- OtherLOC: geographic locations other than settlements (mountains, rivers, oceans, continents, countries, regions, planets)\n"
        "- Facility: specific named built structures (airports, stadiums, museums, hospitals, hotels, bridges, parks)\n"
        "- Station: train/subway/bus/space stations, airports by station name\n"
        "\nPRODUCT/WORK types:\n"
        "- ArtWork: specific art pieces (paintings, sculptures, named artworks)\n"
        "- Clothing: clothing items, fashion brands, specific garments\n"
        "- Drink: specific beverages, brands, types of alcohol\n"
        "- Food: specific foods, dishes, cuisine brands, recipes\n"
        "- OtherPROD: products not fitting specific categories (electronic devices, tools, machinery)\n"
        "- Software: software applications, operating systems, video games, platforms, apps\n"
        "- Vehicle: specific vehicle models or named vehicles (Boeing 737, RMS Titanic, Toyota Camry)\n"
        "- MusicalWork: songs, albums, musical compositions, operas\n"
        "- VisualWork: films, TV shows, documentaries, anime (named visual creative works)\n"
        "- WrittenWork: books, novels, newspapers, magazines, poems, scripts, articles\n"
        "\nMEDICAL types:\n"
        "- AnatomicalStructure: body parts, organs, tissues, anatomical regions, biological structures\n"
        "- Disease: medical conditions, illnesses, syndromes, disorders (COVID-19, diabetes, cancer)\n"
        "- MedicalProcedure: surgical operations, medical treatments, clinical procedures (appendectomy, chemotherapy)\n"
        "- Medication/Vaccine: drugs, medicines, vaccines, pharmaceutical products\n"
        "- Symptom: medical symptoms, clinical signs, patient-reported manifestations\n"
        "\n=== CRITICAL DISAMBIGUATION GUIDE (most common confusion pairs) ===\n"
        "PERSON subtypes — determine by PRIMARY profession/role:\n"
        "  • Artist vs OtherPER: Artist = MUST have a visual/performing art focus "
        "(painter, sculptor, actor, filmmaker, photographer, illustrator, choreographer). "
        "Writer/novelist/poet → WrittenWork-author = OtherPER. Musician → MusicalGRP-member = OtherPER unless specifically a performer.\n"
        "  • Politician vs OtherPER: Politician = MUST have held or currently holds elected/appointed government office "
        "(president, senator, minister, mayor, governor, MP). Political activists without office → OtherPER.\n"
        "  • Scientist vs OtherPER: Scientist = primarily known for research/academic/scientific work "
        "(professor, researcher, inventor, mathematician, naturalist, astronomer). Doctors in research = Scientist; doctors in practice alone = OtherPER.\n"
        "  • Athlete vs OtherPER: Athlete = active sports competitor/player. Retired athlete = still Athlete.\n"
        "  • SportsManager vs Athlete: SportsManager = COACH or team manager. Player = Athlete.\n"
        "  FALLBACK RULE for person types: If genuinely uncertain between specific subtype and OtherPER → use OtherPER. "
        "Never guess Artist when uncertain.\n"
        "LOCATION subtypes — KEY RULE:\n"
        "  • HumanSettlement: ONLY human-inhabited places — cities, towns, villages, neighborhoods, districts, boroughs.\n"
        "  • OtherLOC: countries, nations, states, continents, oceans, seas, rivers, mountains, regions, planets.\n"
        "  CRITICAL: Countries and nations → ALWAYS OtherLOC, NEVER HumanSettlement. "
        "'United States', 'Germany', 'Iran', 'Britain' are ALL OtherLOC.\n"
        "ORGANIZATION subtypes:\n"
        "  • PublicCorp: must be verifiably listed on a stock exchange (NYSE, NASDAQ, etc.)\n"
        "  • PrivateCorp: clearly a private company; when uncertain between public/private → use ORG\n"
        "  • ORG: use as fallback for any organization that does not clearly fit other org subtypes\n"
        "\nNONE = this span is NOT a named entity (common noun, adjective, verb, or generic phrase).\n"
        "When unsure between two types, choose the more specific one — except for persons, where OtherPER is the safe fallback.\n"
    )


def _build_multiconer_type_rules() -> str:
    return (
        "Type definitions for MultiCoNER v2 (33 fine-grained types):\n"
        "\nPERSON types:\n"
        "- OtherPER: persons not fitting specific categories (writers, philosophers, historical figures, unnamed individuals)\n"
        "- Artist: visual artists, painters, sculptors, illustrators, photographers, directors, choreographers\n"
        "- Athlete: sports players, olympians, competitors in any sport\n"
        "- Cleric: religious leaders, priests, popes, imams, monks, rabbis, pastors\n"
        "- Politician: government officials, presidents, prime ministers, senators, mayors, political leaders\n"
        "- Scientist: researchers, academics, engineers, inventors, mathematicians, doctors (named scientists)\n"
        "- SportsManager: sports coaches, team managers, athletic directors, sports executives\n"
        "\nORGANIZATION types:\n"
        "- ORG: general organizations, NGOs, government bodies, universities, schools, hospitals (not fitting below)\n"
        "- AerospaceManufacturer: aerospace/aviation companies and agencies (Boeing, Airbus, SpaceX, NASA programs)\n"
        "- CarManufacturer: automotive companies (Toyota, Ford, Tesla, BMW, Honda)\n"
        "- MusicalGRP: bands, orchestras, choirs, music groups (Beatles, Vienna Philharmonic)\n"
        "- PrivateCorp: privately held companies (not aerospace/auto/musical)\n"
        "- PublicCorp: publicly traded corporations on stock markets\n"
        "- SportsGRP: sports teams, leagues, federations, tournaments as entities (NFL, Premier League, Manchester United)\n"
        "\nLOCATION types:\n"
        "- HumanSettlement: cities, towns, villages, boroughs, districts, neighborhoods\n"
        "- OtherLOC: geographic locations other than settlements (mountains, rivers, oceans, continents, countries, regions, planets)\n"
        "- Facility: specific named built structures (airports, stadiums, museums, hospitals, hotels, bridges, parks)\n"
        "- Station: train/subway/bus/space stations, airports by station name\n"
        "\nPRODUCT/WORK types:\n"
        "- ArtWork: specific art pieces (paintings, sculptures, named artworks)\n"
        "- Clothing: clothing items, fashion brands, specific garments\n"
        "- Drink: specific beverages, brands, types of alcohol\n"
        "- Food: specific foods, dishes, cuisine brands, recipes\n"
        "- OtherPROD: products not fitting specific categories (electronic devices, tools, machinery)\n"
        "- Software: software applications, operating systems, video games, platforms, apps\n"
        "- Vehicle: specific vehicle models or named vehicles (Boeing 737, RMS Titanic, Toyota Camry)\n"
        "- MusicalWork: songs, albums, musical compositions, operas\n"
        "- VisualWork: films, TV shows, documentaries, anime (named visual creative works)\n"
        "- WrittenWork: books, novels, newspapers, magazines, poems, scripts, articles\n"
        "\nMEDICAL types:\n"
        "- AnatomicalStructure: body parts, organs, tissues, anatomical regions, biological structures\n"
        "- Disease: medical conditions, illnesses, syndromes, disorders (COVID-19, diabetes, cancer)\n"
        "- MedicalProcedure: surgical operations, medical treatments, clinical procedures (appendectomy, chemotherapy)\n"
        "- Medication/Vaccine: drugs, medicines, vaccines, pharmaceutical products\n"
        "- Symptom: medical symptoms, clinical signs, patient-reported manifestations\n"
        "\nNONE = this span is NOT a named entity (common noun, adjective, verb, or generic phrase).\n"
        "When unsure between two types, choose the more specific one.\n"
    )


def build_type_prompt(
    trans_text: str,
    mentions: List[Dict],
    language: str,
    label_schema: List[str],
    use_cot: bool = False,
) -> str:
    labels = ", ".join(label_schema)
    type_options = "|".join(label_schema) + "|NONE"
    mentions_json = json.dumps(mentions, ensure_ascii=False)

    lang_code = str(language).strip().lower()
    lang_name = _LANG_NAMES.get(lang_code, lang_code.upper())
    is_non_latin = lang_code in _NON_LATIN
    is_low_resource = lang_code in _LOW_RESOURCE

    # Language-aware instruction for TypeAgent
    if lang_code == "en":
        lang_hint = ""  # English: mention text IS English, no special hint needed
    elif is_non_latin:
        lang_hint = (
            f"NOTE: The original language is {lang_name} (non-Latin script). "
            "The candidate span text will NOT look like English. "
            "Rely primarily on the English translation to identify what each span refers to.\n"
        )
    elif is_low_resource:
        lang_hint = (
            f"NOTE: The original language is {lang_name} (low-resource). "
            "The English translation may be imperfect; use your best judgment.\n"
        )
    else:
        lang_hint = f"NOTE: The original language is {lang_name}.\n"

    # Schema-specific type definitions
    schema_upper = [x.upper() for x in label_schema]
    has_date = "DATE" in schema_upper
    is_wikiann = set(schema_upper) <= {"PER", "ORG", "LOC", "DATE"}
    is_misc = "MISC" in schema_upper

    if is_wikiann:
        date_rule = (
            "DATE=specific dates, times, years, periods, or temporal expressions "
            "(e.g., 'Monday', 'January 1st', '2023', 'last year', 'World War II era'). "
            "DATE only applies when a temporal expression is an actual named entity in context. "
            "When unsure between DATE and other types → prefer ORG or PER.\n"
            if has_date else ""
        )
        rule_types = (
            "Type definitions: PER=persons/humans only, "
            "LOC=geographic places only (cities, countries, regions, rivers, mountains), "
            "ORG=everything else named (companies, institutions, teams, facilities, buildings, "
            "events, works, medical terms, products, etc). When unsure between LOC and ORG → prefer ORG.\n"
            + date_rule
        )
    elif is_misc:
        rule_types = (
            "Type definitions:\n"
            "- PER: individual persons by name (e.g., 'John Smith', 'García')\n"
            "- ORG: SPECIFIC named organizations, companies, institutions, sports teams/leagues "
            "(must be a proper noun referring to a named entity — NOT generic words like "
            "'construction', 'development', 'international', 'first', 'national', 'tournament')\n"
            "- LOC: geographic places (countries, cities, regions, rivers, mountains)\n"
            "- MISC: everything else — IMPORTANT: this INCLUDES:\n"
            "  * Nationality/ethnic/religious/political descriptors: 'European', 'German', 'Thai', "
            "'Muslim', 'Jewish', 'Serbian', 'Dutch', 'Hungarian', 'Social Democrats', 'conservatives'\n"
            "  * Disease/medical terms: 'BSE', 'encephalopathy', 'bovine spongiform encephalopathy'\n"
            "  * Event names: 'World Cup', 'Olympics', 'Grammy Award'\n"
            "  * International agreements/frameworks: 'GATT', 'NATO'\n"
            "  * Products, works, abbreviations that don't fit PER/ORG/LOC\n"
            "CRITICAL: Nationality adjectives ('European', 'Hungarian', 'Dutch') ARE valid MISC entities "
            "even when used as adjectives modifying other nouns.\n"
        )
    elif _is_multiconer_schema(schema_upper):
        rule_types = _build_multiconer_type_rules_v2() if use_cot else _build_multiconer_type_rules()
    else:
        rule_types = "Assign the most specific matching type from the list above.\n"

    is_multiconer = _is_multiconer_schema(schema_upper)
    if is_misc:
        rule_types += (
            "NONE = this span is a common word, verb, generic noun, or pure adjective with NO named-entity meaning.\n"
            "Do NOT use NONE for nationality words, disease names, event names, or political group names.\n"
        )
    elif not is_multiconer:
        # MC2 rules already include NONE; other schemas need the generic NONE
        rule_types += "NONE = this span is NOT a named entity (common noun, adjective, verb, or other non-entity).\n"

    rule3 = (
        "3) Be precise: reject generic common nouns and verbs. "
        + ("Exception: nationality words and ethnic/religious descriptors ARE entities (→ MISC) even as adjectives.\n"
           if is_misc else "Choose the most specific type that fits.\n")
    )

    cot_instruction = ""
    cot_format_field = f'"type": "{type_options}", "confidence": 0.9, "reason": "..."'
    if use_cot and _is_multiconer_schema(schema_upper):
        cot_instruction = (
            "STEP-BY-STEP REASONING: For each span, first write a brief reasoning "
            "(one sentence describing what the entity is and which category it belongs to), "
            "then assign the type. Include a 'reasoning' field in your JSON output.\n"
        )
        cot_format_field = (
            f'"reasoning": "one-sentence entity description → category decision", '
            f'"type": "{type_options}", "confidence": 0.9, "reason": "..."'
        )

    return (
        "You are TypeAgent in KARVE.\n"
        "Task: For each candidate span, FIRST decide if it is a true named entity, THEN classify its type.\n"
        "The spans were extracted from the original-language text; use the English translation to verify and classify.\n"
        f"{lang_hint}"
        f"{cot_instruction}"
        "Rules:\n"
        f"1) Valid types: {labels}. If the span is NOT a named entity → use NONE.\n"
        f"2) {rule_types}"
        f"{rule3}"
        "4) Use the English translation to understand what each span refers to. "
        "If a mention has an 'en_context' field, it is the likely English equivalent — use it.\n"
        "5) Output strict JSON only.\n\n"
        f"Original language: {lang_name}\n"
        f"English translation: {trans_text}\n"
        f"Candidate spans to verify and classify: {mentions_json}\n\n"
        "Return format:\n"
        "{\n"
        "  \"classifications\": [\n"
        f"    {{\"id\": 0, {cot_format_field}}}\n"
        "  ]\n"
        "}\n"
    )
