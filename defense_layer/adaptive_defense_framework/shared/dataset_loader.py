import ast
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional

from .schema import DialogueSample, Turn
from .text_features import normalize_text


def _safe_parse_list(serialized: str):
    if not serialized:
        return []
    for fn in (json.loads, ast.literal_eval):
        try:
            parsed = fn(serialized)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            continue
    return []


def _to_turns_from_history(history) -> List[Turn]:
    turns: List[Turn] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        # SafeDialBench style
        if "user" in item:
            turns.append(Turn(role="user", content=normalize_text(item.get("user", ""))))
        if "bot" in item:
            turns.append(Turn(role="assistant", content=normalize_text(item.get("bot", ""))))
        # MHJ fallback style
        if "role" in item and "content" in item:
            role = "assistant" if item["role"] == "assistant" else "user"
            content = normalize_text(item.get("content", ""))
            if content.lower() == "none":
                content = ""
            turns.append(Turn(role=role, content=content))
    # remove empty assistant placeholders
    return [t for t in turns if not (t.role == "assistant" and not t.content)]


def _limit(samples: List[DialogueSample], limit: Optional[int]) -> List[DialogueSample]:
    if limit is None or limit < 0:
        return samples
    return samples[:limit]


def load_safedialbench_samples(
    dataset_root: str,
    language: str = "en",
    limit: Optional[int] = None,
) -> List[DialogueSample]:
    root = Path(dataset_root)
    fn = "datasets_en.jsonl" if language.lower() == "en" else "datasets_zh.jsonl"
    path = root / "SafeDialBench" / fn
    samples: List[DialogueSample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            turns = _to_turns_from_history(obj.get("history", []))
            samples.append(
                DialogueSample(
                    sample_id=f"safedial_{obj.get('id', idx)}",
                    source="SafeDialBench",
                    # SafeDialBench entries here are attack-context dialogues.
                    label=1,
                    turns=turns,
                    meta={
                        "task": obj.get("task"),
                        "method": obj.get("method"),
                        "scene": obj.get("scene"),
                        "model_type": obj.get("model_type"),
                    },
                )
            )
    return _limit(samples, limit)


def _read_csv_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_mhj_fallback_samples(
    dataset_root: str,
    harmful_limit: Optional[int] = None,
    benign_limit: Optional[int] = None,
) -> List[DialogueSample]:
    root = Path(dataset_root) / "MHJ_fallback_public"
    harmful_path = root / "Harmful_Dataset.csv"
    benign_paths = [
        root / "Completely-Benign_Dataset.csv",
        root / "Semi-Benign_Dataset.csv",
    ]

    samples: List[DialogueSample] = []

    harmful_count = 0
    for i, row in enumerate(_read_csv_rows(harmful_path)):
        if harmful_limit is not None and harmful_limit >= 0 and harmful_count >= harmful_limit:
            break
        turns = _to_turns_from_history(_safe_parse_list(row.get("Multi-turn conversation", "")))
        samples.append(
            DialogueSample(
                sample_id=f"mhj_h_{i}",
                source="MHJ_fallback_public",
                label=1,
                turns=turns,
                meta={
                    "goal_id": row.get("Goal ID"),
                    "goal": row.get("Goal"),
                    "input_cipher": row.get("Input-cipher"),
                    "output_cipher": row.get("Output-cipher"),
                },
            )
        )
        harmful_count += 1

    benign_count = 0
    per_file_limit = None
    if benign_limit is not None and benign_limit >= 0:
        per_file_limit = max(1, benign_limit // max(1, len(benign_paths)))

    for path in benign_paths:
        local_count = 0
        for i, row in enumerate(_read_csv_rows(path)):
            if per_file_limit is not None and local_count >= per_file_limit:
                break
            turns = _to_turns_from_history(_safe_parse_list(row.get("Multi-turn conversation", "")))
            samples.append(
                DialogueSample(
                    sample_id=f"mhj_b_{path.stem}_{i}",
                    source="MHJ_fallback_public",
                    label=0,
                    turns=turns,
                    meta={
                        "goal_id": row.get("Goal ID"),
                        "goal": row.get("Goal"),
                        "input_cipher": row.get("Input-cipher"),
                        "output_cipher": row.get("Output-cipher"),
                        "subset": path.stem,
                    },
                )
            )
            benign_count += 1
            local_count += 1
            if benign_limit is not None and benign_limit >= 0 and benign_count >= benign_limit:
                break
        if benign_limit is not None and benign_limit >= 0 and benign_count >= benign_limit:
            break

    return samples


def load_mhj_completely_benign_samples(
    dataset_root: str,
    limit: Optional[int] = None,
) -> List[DialogueSample]:
    root = Path(dataset_root) / "MHJ_fallback_public"
    path = root / "Completely-Benign_Dataset.csv"
    samples: List[DialogueSample] = []
    for i, row in enumerate(_read_csv_rows(path)):
        if limit is not None and limit >= 0 and i >= limit:
            break
        turns = _to_turns_from_history(_safe_parse_list(row.get("Multi-turn conversation", "")))
        samples.append(
            DialogueSample(
                sample_id=f"mhj_cb_{i}",
                source="MHJ_fallback_public",
                label=0,
                turns=turns,
                meta={
                    "goal_id": row.get("Goal ID"),
                    "goal": row.get("Goal"),
                    "input_cipher": row.get("Input-cipher"),
                    "output_cipher": row.get("Output-cipher"),
                    "subset": "Completely-Benign",
                },
            )
        )
    return samples


def load_local_safe_dialogs(jsonl_path: str, limit: Optional[int] = None) -> List[DialogueSample]:
    path = Path(jsonl_path)
    samples: List[DialogueSample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and limit >= 0 and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            history = obj.get("history", [])
            turns = _to_turns_from_history(history)
            samples.append(
                DialogueSample(
                    sample_id=str(obj.get("id", f"safe_{i}")),
                    source="LocalSafeDialogs",
                    label=0,
                    turns=turns,
                    meta={"topic": obj.get("topic", "general")},
                )
            )
    return samples
