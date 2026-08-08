from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple, Type, Union


class JsonSchemaChecker:
    """Lightweight schema checker for JSON-like lists."""

    def __init__(self, schema: Dict[str, Union[Type, Tuple[Type, ...]]]):
        self.schema = schema

    def check(self, data: Union[str, list]) -> bool:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return False
        if not isinstance(data, list):
            return False
        return all(self._check_item(item) for item in data)

    def _check_item(self, item: Dict[str, Any]) -> bool:
        if not isinstance(item, dict):
            return False
        for k, t in self.schema.items():
            if k not in item or not isinstance(item[k], t):
                return False
        return True


def _strip_response_text(text: str) -> str:
    """Strip markdown code blocks and model-specific prefixes (think tags, etc.)."""
    # Remove Qwen/DeepSeek-R1 thinking blocks: <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Remove markdown code fences: ```json\n...\n``` or ```\n...\n```
    m = re.match(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


def parse_json(text: str, default: Any = None) -> Any:
    text = _strip_response_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find first JSON object or array in the text
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return default
