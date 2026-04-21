import re
import unicodedata


ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
SPECIAL_PATTERN = re.compile(r"[^\w\s\u4e00-\u9fff\.,;:!?\-_/()'\"#]+")


def sanitize_user_input(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    normalized = ZERO_WIDTH_PATTERN.sub("", normalized)
    normalized = CONTROL_PATTERN.sub(" ", normalized)
    normalized = normalized.replace("\\n", " ").replace("\\t", " ")
    normalized = SPECIAL_PATTERN.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    return normalized
