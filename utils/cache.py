from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class SimpleFileCache:
    """Simple file-based JSON cache for deterministic experiment utilities."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        payload = {
            "created_at": int(time.time()),
            "ttl_seconds": ttl_seconds,
            "value": value,
        }
        path = self._key_path(key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get(self, key: str) -> Any:
        path = self._key_path(key)
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        ttl = payload.get("ttl_seconds")
        created = int(payload.get("created_at", 0))
        if ttl is not None and (int(time.time()) - created) > int(ttl):
            try:
                path.unlink()
            except OSError:
                pass
            return None

        return payload.get("value")
