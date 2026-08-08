from __future__ import annotations

import random
from typing import Dict, List, Sequence


def low_resource_sample(items: Sequence[Dict], ratio: float, seed: int = 42) -> List[Dict]:
    if ratio >= 1.0:
        return list(items)
    if ratio <= 0.0:
        return []
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    k = max(1, int(len(items) * ratio))
    return [items[i] for i in idx[:k]]
