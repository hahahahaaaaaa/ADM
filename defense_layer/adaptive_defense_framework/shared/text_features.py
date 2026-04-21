import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("\u3000", " ").split())


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text).lower())


def hash_embed(text: str, dim: int = 1024) -> List[float]:
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    counts = Counter(tokens)
    for tok, cnt in counts.items():
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += float(cnt)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def keyword_score(text: str, keyword_weights: Dict[str, float]) -> float:
    lowered = normalize_text(text).lower()
    score = 0.0
    for kw, w in keyword_weights.items():
        if kw in lowered:
            score += w
    return score


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
