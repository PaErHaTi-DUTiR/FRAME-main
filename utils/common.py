from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from .json_parser import JsonSchemaChecker


def read_json(path: Union[str, Path]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
