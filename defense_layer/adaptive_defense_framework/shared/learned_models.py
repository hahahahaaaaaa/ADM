import ast
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from ..module_3_adaptive_defense_policy.input_guard import sanitize_user_input
from .text_features import normalize_text


BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = ARTIFACT_DIR / "learned_defense_bundle_v2.joblib"

ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MHJ_HARMFUL_PATH = PROJECT_ROOT / "datasets" / "MHJ_fallback_public" / "Harmful_Dataset.csv"
MHJ_BENIGN_PATH = PROJECT_ROOT / "datasets" / "MHJ_fallback_public" / "Completely-Benign_Dataset.csv"
MHJ_SEMIBENIGN_PATH = PROJECT_ROOT / "datasets" / "MHJ_fallback_public" / "Semi-Benign_Dataset.csv"
SAFEDIAL_EN_PATH = PROJECT_ROOT / "datasets" / "SafeDialBench" / "datasets_en.jsonl"
SAFEDIAL_ZH_PATH = PROJECT_ROOT / "datasets" / "SafeDialBench" / "datasets_zh.jsonl"
SAFE_DIALOGS_PATH = PROJECT_ROOT / "interfaces" / "safe_test_dialogs.jsonl"
DEMO_CASES_PATH = PROJECT_ROOT / "interfaces" / "demo_cases.json"
SAFE50_QR_PATH = PROJECT_ROOT / "results" / "evaluation_safe50" / "safe50_qr.jsonl"
HARMFUL_BEHAVIORS_PATH = PROJECT_ROOT / "datasets" / "external" / "harmful_behaviors.csv"

MAX_SAMPLES_PER_SPLIT = 6000


def _dedupe_limit(texts: Sequence[str], limit: int = MAX_SAMPLES_PER_SPLIT) -> List[str]:
    cleaned = []
    seen = set()
    for text in texts:
        norm = sanitize_user_input(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(norm)
        if len(cleaned) >= limit:
            break
    return cleaned


def _parse_history_literal(history_text: str) -> List[Dict]:
    try:
        return ast.literal_eval(history_text)
    except Exception:
        return []


def _collect_mhj_texts(csv_path: Path) -> Tuple[List[str], List[str]]:
    user_turns: List[str] = []
    assistant_turns: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_turns.append(row.get("Goal", ""))
            for item in _parse_history_literal(row.get("Multi-turn conversation", "")):
                role = item.get("role")
                content = item.get("content")
                if content in {None, "", "None"}:
                    continue
                if role == "user":
                    user_turns.append(content)
                elif role == "assistant":
                    assistant_turns.append(content)
    return user_turns, assistant_turns


def _collect_safedial_texts(jsonl_path: Path) -> Dict[str, List[str]]:
    user_first: List[str] = []
    user_last: List[str] = []
    bot_first: List[str] = []
    bot_last: List[str] = []
    all_user: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_turns = []
            bot_turns = []
            for item in row.get("history", []):
                user_text = item.get("user", "")
                bot_text = item.get("bot", "")
                if normalize_text(user_text):
                    user_turns.append(user_text)
                    all_user.append(user_text)
                if normalize_text(bot_text):
                    bot_turns.append(bot_text)
            if user_turns:
                user_first.append(user_turns[0])
                user_last.append(user_turns[-1])
            if bot_turns:
                bot_first.append(bot_turns[0])
                bot_last.append(bot_turns[-1])
    return {
        "user_first": user_first,
        "user_last": user_last,
        "bot_first": bot_first,
        "bot_last": bot_last,
        "all_user": all_user,
    }


def _collect_safe_dialog_texts() -> Tuple[List[str], List[str]]:
    user_texts: List[str] = []
    bot_texts: List[str] = []
    with SAFE_DIALOGS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for item in row.get("history", []):
                user_texts.append(item.get("user", ""))
                bot_texts.append(item.get("bot", ""))
    if DEMO_CASES_PATH.exists():
        demo_cases = json.loads(DEMO_CASES_PATH.read_text(encoding="utf-8"))
        for item in demo_cases:
            for turn in item.get("turns", []):
                if turn.get("role") == "user":
                    user_texts.append(turn.get("content", ""))
                elif turn.get("role") == "assistant":
                    bot_texts.append(turn.get("content", ""))
    if SAFE50_QR_PATH.exists():
        with SAFE50_QR_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                user_texts.append(row.get("question", ""))
                bot_texts.append(row.get("response", ""))
    return user_texts, bot_texts


def _collect_harmful_behavior_targets() -> Tuple[List[str], List[str]]:
    user_texts: List[str] = []
    bot_texts: List[str] = []
    if not HARMFUL_BEHAVIORS_PATH.exists():
        return user_texts, bot_texts
    with HARMFUL_BEHAVIORS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_texts.append(row.get("Goal", ""))
            bot_texts.append(row.get("target", ""))
    return user_texts, bot_texts


def _sample_for_centroid(texts: Sequence[str], limit: int = 2000) -> List[str]:
    return list(texts[:limit])


def _train_classifier(encoder: SentenceTransformer, positive_texts: Sequence[str], negative_texts: Sequence[str]) -> LogisticRegression:
    pos = _dedupe_limit(positive_texts)
    neg = _dedupe_limit(negative_texts)
    texts = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=7,
    )
    clf.fit(embeddings, labels)
    return clf


def _fit_bundle() -> Dict:
    encoder = SentenceTransformer(ENCODER_NAME, device="cpu")

    mhj_user_pos, _ = _collect_mhj_texts(MHJ_HARMFUL_PATH)
    mhj_user_neg, _ = _collect_mhj_texts(MHJ_BENIGN_PATH)
    mhj_semi_user_neg, _ = _collect_mhj_texts(MHJ_SEMIBENIGN_PATH)
    safedial_en = _collect_safedial_texts(SAFEDIAL_EN_PATH)
    safedial_zh = _collect_safedial_texts(SAFEDIAL_ZH_PATH)
    safe_user, safe_bot = _collect_safe_dialog_texts()
    harmful_user, harmful_bot = _collect_harmful_behavior_targets()

    forward_positive = (
        mhj_user_pos
        + harmful_user
        + safedial_en["user_last"]
        + safedial_zh["user_last"]
        + safedial_en["all_user"][-800:]
        + safedial_zh["all_user"][-800:]
    )
    forward_negative = (
        mhj_user_neg
        + mhj_semi_user_neg
        + safe_user
        + safedial_en["user_first"]
        + safedial_zh["user_first"]
    )

    backward_positive = harmful_bot + safedial_en["bot_last"] + safedial_zh["bot_last"]
    backward_negative = safe_bot + safedial_en["bot_first"] + safedial_zh["bot_first"]

    forward_clf = _train_classifier(encoder, forward_positive, forward_negative)
    backward_clf = _train_classifier(encoder, backward_positive, backward_negative)

    safe_anchor_texts = _sample_for_centroid(_dedupe_limit(forward_negative, limit=2500))
    harmful_anchor_texts = _sample_for_centroid(_dedupe_limit(forward_positive, limit=2500))

    safe_centroid = encoder.encode(
        safe_anchor_texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).mean(axis=0)
    harmful_centroid = encoder.encode(
        harmful_anchor_texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).mean(axis=0)
    return {
        "version": "v2",
        "forward_classifier": forward_clf,
        "backward_classifier": backward_clf,
        "safe_centroid": safe_centroid,
        "harmful_centroid": harmful_centroid,
        "metadata": {
            "encoder_name": ENCODER_NAME,
            "forward_positive_count": len(_dedupe_limit(forward_positive)),
            "forward_negative_count": len(_dedupe_limit(forward_negative)),
            "backward_positive_count": len(_dedupe_limit(backward_positive)),
            "backward_negative_count": len(_dedupe_limit(backward_negative)),
        },
    }


class LearnedDefenseModels:
    def __init__(self):
        self.encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
        if ARTIFACT_PATH.exists():
            bundle = joblib.load(ARTIFACT_PATH)
        else:
            bundle = _fit_bundle()
            joblib.dump(bundle, ARTIFACT_PATH)
        self.forward_classifier = bundle["forward_classifier"]
        self.backward_classifier = bundle["backward_classifier"]
        self.safe_centroid = bundle["safe_centroid"]
        self.harmful_centroid = bundle["harmful_centroid"]
        self.metadata = bundle["metadata"]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized = [sanitize_user_input(text) for text in texts]
        return self.encoder.encode(
            normalized,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def forward_score(self, text: str) -> float:
        if not normalize_text(text):
            return 0.0
        emb = self.encode([text])
        return float(self.forward_classifier.predict_proba(emb)[0][1])

    def backward_score(self, text: str) -> float:
        if not normalize_text(text):
            return 0.0
        emb = self.encode([text])
        return float(self.backward_classifier.predict_proba(emb)[0][1])


@lru_cache(maxsize=1)
def get_learned_models() -> LearnedDefenseModels:
    return LearnedDefenseModels()




