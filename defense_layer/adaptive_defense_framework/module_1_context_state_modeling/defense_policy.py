from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

from ..shared.text_features import normalize_text


@dataclass
class DefenseDecision:
    action: str
    risk_level: str
    message: str
    strategy: str = ""
    safety_notice: str = ""
    preserved_excerpt: str = ""
    removed_fragments_count: int = 0


class AdaptiveDefensePolicy:
    # Tiered control: allow -> guide -> rewrite -> partial_refuse -> refuse
    def __init__(self):
        self.level_thresholds = {
            "low": 0.18,
            "medium": 0.48,
            "high": 0.66,
        }
        self._sensitive_patterns = [
            r"\b(step[- ]?by[- ]?step|instructions?|procedure|workflow)\b",
            r"\b(make|build|create|develop|deploy|execute)\b",
            r"\b(bypass|exploit|jailbreak|ignore safety|disable guardrail)\b",
            r"\b(payload|script|command|parameter|materials?)\b",
        ]
        self._procedural_patterns = [
            r"(^|\n)\s*(\d+[\.\)]|[-*])\s+",
            r"\b(first|second|third|then|after that|next)\b",
            r"\b(exactly|specifically|detailed?|precise)\b",
        ]
        self._category_patterns = {
            "cyber": r"\b(hack|hacking|exploit|payload|malware|virus|phishing|credential|network|database|shell|sql|xss)\b",
            "violence": r"\b(bomb|weapon|firearm|poison|harm|kill|assassinate|explosive)\b",
            "fraud": r"\b(identity theft|fraud|scam|steal|money laundering|manipulate stock|counterfeit)\b",
            "harassment": r"\b(racism|harass|body shaming|demean|defamatory|violence against|doxx)\b",
        }
        self._parameter_patterns = [
            re.compile(r"\b\d{2,}\b"),
            re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            re.compile(r"https?://\S+"),
            re.compile(r"\b[a-f0-9]{16,}\b", re.IGNORECASE),
            re.compile(r"`[^`]+`"),
        ]
        self._risky_term_map = {
            "bypass": "相关绕过表述",
            "exploit": "相关利用表述",
            "payload": "相关载荷表述",
            "script": "相关脚本表述",
            "command": "相关命令表述",
            "step by step": "相关分步说明",
            "instructions": "相关说明",
        }
        self._guide_whitelist_patterns = [
            re.compile(r"(translate|translation|翻译|译成|译为)", re.IGNORECASE),
            re.compile(r"(summari[sz]e|summary|概括|总结|摘要)", re.IGNORECASE),
            re.compile(r"(extract|extraction|提取|抽取|信息抽取)", re.IGNORECASE),
            re.compile(r"(sentiment|情感|倾向)", re.IGNORECASE),
            re.compile(r"(explain.*code|code explanation|解释.*代码|解释.*python|解释.*表达式)", re.IGNORECASE),
            re.compile(r"(qa|question answer|常规问答|简要回答|什么是)", re.IGNORECASE),
        ]
        self._single_turn_harmful_patterns = [
            re.compile(r"\b(hack|malware|virus|phishing|credential|explosive|weapon|poison)\b", re.IGNORECASE),
            re.compile(r"\b(fraud|scam|steal|counterfeit|launder|identity theft)\b", re.IGNORECASE),
            re.compile(r"\b(defam\w*|slander|harass\w*|demean|body shaming|xenophob\w*)\b", re.IGNORECASE),
        ]
        self._single_turn_request_patterns = [
            re.compile(r"\b(how to|ways to|show me how to|tell me how to)\b", re.IGNORECASE),
            re.compile(r"\b(write|draft|generate|create|give me|provide)\b", re.IGNORECASE),
            re.compile(r"\b(plan|guide|manual|instructions?|steps?|procedure)\b", re.IGNORECASE),
        ]

    def level_of(self, risk_score: float) -> str:
        if risk_score < self.level_thresholds["low"]:
            return "low"
        if risk_score < self.level_thresholds["medium"]:
            return "medium"
        if risk_score < self.level_thresholds["high"]:
            return "high"
        return "critical"

    def decide(self, risk_score: float, user_text: str, assistant_text: str) -> DefenseDecision:
        if self._is_refusal_response(assistant_text):
            return self._allow(
                assistant_text,
                "low",
                strategy="preserve_target_refusal",
                notice="目标模型已拒答，直接保留原始安全回复。",
            )

        level = self.level_of(risk_score)
        explicit_single_turn = self._is_explicit_single_turn_request(user_text)
        if explicit_single_turn and level in {"low", "medium"}:
            return self._strong_single_turn_rewrite(user_text, assistant_text, "high")

        escalation = self._single_turn_escalation_level(user_text, assistant_text, risk_score)
        if escalation == "high" and level == "medium":
            return self._sanitize_response(user_text, assistant_text, "high")
        if escalation == "critical" and level in {"medium", "high"}:
            safe_text, removed_count = self._extract_safe_information(assistant_text, aggressive=True)
            if safe_text:
                return self._partial_refusal(user_text, safe_text, removed_count, "critical")
            return self._refusal(user_text, "critical")

        if level == "low":
            return self._allow(assistant_text, level, strategy="直接放行")

        if level == "medium":
            if self._is_benign_whitelisted(user_text, assistant_text):
                return self._allow(
                    assistant_text,
                    level,
                    strategy="白名单放行",
                    notice="检测到通用任务类型，保持原始回答风格。",
                )
            return self._guidance(user_text, assistant_text, level)

        if level == "high":
            return self._sanitize_response(user_text, assistant_text, level)

        safe_text, removed_count = self._extract_safe_information(assistant_text, aggressive=True)
        if safe_text:
            return self._partial_refusal(user_text, safe_text, removed_count, level)
        return self._refusal(user_text, level)

    def _allow(self, assistant_text: str, level: str, strategy: str, notice: str = "") -> DefenseDecision:
        return DefenseDecision(
            action="allow",
            risk_level=level,
            message=assistant_text,
            strategy=strategy,
            safety_notice=notice,
        )

    def _guidance(self, user_text: str, assistant_text: str, level: str) -> DefenseDecision:
        category = self._detect_category(f"{user_text}\n{assistant_text}")
        safe_text, removed_count = self._extract_safe_information(assistant_text, aggressive=False)
        notice = "提示：请仅在合规场景下参考下述信息。"

        if removed_count == 0 and normalize_text(assistant_text):
            message = f"{assistant_text}\n\n{notice}"
            preserved = assistant_text
        else:
            preserved = safe_text or ""
            parts: List[str] = []
            if preserved:
                parts.append(preserved)
            parts.append(notice)
            parts.append(self._safe_redirect(category, compact=True))
            message = "\n\n".join(parts)

        return DefenseDecision(
            action="guide",
            risk_level=level,
            message=message,
            strategy="引导优先",
            safety_notice=notice,
            preserved_excerpt=preserved[:200],
            removed_fragments_count=removed_count,
        )

    def _sanitize_response(self, user_text: str, assistant_text: str, level: str) -> DefenseDecision:
        category = self._detect_category(f"{user_text}\n{assistant_text}")
        safe_text, removed_count = self._extract_safe_information(assistant_text, aggressive=True)
        message = self._build_safe_substitute(category, safe_text=safe_text, compact=False)

        return DefenseDecision(
            action="rewrite",
            risk_level=level,
            message=message,
            strategy="输出侧重写",
            safety_notice="已替换为安全替代内容。",
            preserved_excerpt=safe_text[:200],
            removed_fragments_count=removed_count,
        )

    def _strong_single_turn_rewrite(self, user_text: str, assistant_text: str, level: str) -> DefenseDecision:
        category = self._detect_category(f"{user_text}\n{assistant_text}")
        safe_text, removed_count = self._extract_safe_information(assistant_text, aggressive=True)
        message = self._build_safe_substitute(category, safe_text="", compact=False)
        return DefenseDecision(
            action="rewrite",
            risk_level=level,
            message=message,
            strategy="单轮强重写",
            safety_notice="已替换为安全替代内容。",
            preserved_excerpt=safe_text[:120],
            removed_fragments_count=max(1, removed_count),
        )

    def _partial_refusal(self, user_text: str, safe_text: str, removed_count: int, level: str) -> DefenseDecision:
        category = self._detect_category(user_text)
        notice = "以下仅保留可直接公开讨论的部分。"
        parts = [safe_text, notice, self._safe_redirect(category, compact=False)]
        return DefenseDecision(
            action="partial_refuse",
            risk_level=level,
            message="\n\n".join(parts),
            strategy="局部拒答",
            safety_notice=notice,
            preserved_excerpt=safe_text[:200],
            removed_fragments_count=removed_count,
        )

    def _refusal(self, user_text: str, level: str) -> DefenseDecision:
        category = self._detect_category(user_text)
        notice = "该请求中的关键部分不适合继续提供。"
        return DefenseDecision(
            action="refuse",
            risk_level=level,
            message=f"{notice}\n\n{self._safe_redirect(category, compact=False)}",
            strategy="拒答兜底",
            safety_notice=notice,
        )

    def _is_benign_whitelisted(self, user_text: str, assistant_text: str) -> bool:
        normalized_user = normalize_text(user_text).lower()
        normalized_assistant = normalize_text(assistant_text).lower()

        if not normalized_user:
            return False

        if any(re.search(pattern, normalized_user, flags=re.IGNORECASE) for pattern in self._sensitive_patterns):
            return False
        if any(re.search(pattern, normalized_assistant, flags=re.IGNORECASE) for pattern in self._sensitive_patterns):
            return False
        if self._detect_category(f"{normalized_user}\n{normalized_assistant}") != "general":
            return False

        has_safe_task_pattern = any(pattern.search(normalized_user) for pattern in self._guide_whitelist_patterns)
        is_short_qa = (
            ("?" in user_text or "？" in user_text)
            and len(normalized_user.split()) <= 14
            and not any(re.search(pattern, normalized_assistant, flags=re.IGNORECASE) for pattern in self._procedural_patterns)
            and "```" not in assistant_text
            and len(normalized_assistant.split()) <= 120
        )
        return has_safe_task_pattern or is_short_qa

    def _is_refusal_response(self, assistant_text: str) -> bool:
        normalized = normalize_text(assistant_text).lower()
        if not normalized:
            return False
        refusal_patterns = [
            re.compile(r"\bcan[^a-z0-9]?t\s+(help|assist|comply|provide)\b", re.IGNORECASE),
            re.compile(r"\bcannot\s+(help|assist|comply|provide)\b", re.IGNORECASE),
            re.compile(r"\bunable\s+to\s+help\b", re.IGNORECASE),
            re.compile(r"\bnot able to (help|comply) with that request\b", re.IGNORECASE),
            re.compile(r"(不能协助|无法协助|不能提供|无法提供|抱歉)", re.IGNORECASE),
        ]
        return any(pattern.search(normalized) for pattern in refusal_patterns)

    def _single_turn_escalation_level(self, user_text: str, assistant_text: str, risk_score: float) -> str:
        normalized_user = normalize_text(user_text).lower()
        if not normalized_user:
            return ""
        harmful_hit = any(pattern.search(normalized_user) for pattern in self._single_turn_harmful_patterns)
        request_hit_count = sum(1 for pattern in self._single_turn_request_patterns if pattern.search(normalized_user))
        procedural_output = any(re.search(pattern, assistant_text, flags=re.IGNORECASE) for pattern in self._procedural_patterns)

        if harmful_hit and request_hit_count >= 2 and (procedural_output or risk_score >= 0.40):
            return "critical"
        if harmful_hit and request_hit_count >= 1 and risk_score >= 0.34:
            return "high"
        return ""

    def _is_explicit_single_turn_request(self, user_text: str) -> bool:
        normalized_user = normalize_text(user_text).lower()
        if not normalized_user:
            return False
        harmful_hit = any(pattern.search(normalized_user) for pattern in self._single_turn_harmful_patterns)
        request_hit_count = sum(1 for pattern in self._single_turn_request_patterns if pattern.search(normalized_user))
        return harmful_hit and request_hit_count >= 1

    def _extract_safe_information(self, assistant_text: str, aggressive: bool) -> Tuple[str, int]:
        text = normalize_text(assistant_text)
        if not text:
            return "", 0

        fragments = self._split_fragments(text)
        kept: List[str] = []
        removed = 0
        for fragment in fragments:
            if self._is_sensitive_fragment(fragment, aggressive=aggressive):
                removed += 1
                continue
            rewritten = self._blur_sensitive_parameters(self._replace_risky_terms(fragment))
            rewritten = normalize_text(rewritten)
            if rewritten:
                kept.append(rewritten)
        return " ".join(kept[:4]), removed

    def _split_fragments(self, text: str) -> List[str]:
        chunks = re.split(r"(?<=[。！？?!])\s+|\n+", text)
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def _is_sensitive_fragment(self, fragment: str, aggressive: bool) -> bool:
        normalized = normalize_text(fragment).lower()
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self._sensitive_patterns):
            return True
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self._procedural_patterns):
            return True
        if aggressive and re.search(r"\b(use|run|set|open|download|install|connect)\b", normalized):
            return True
        return False

    def _blur_sensitive_parameters(self, text: str) -> str:
        rewritten = text
        for pattern in self._parameter_patterns:
            rewritten = pattern.sub("[已模糊参数]", rewritten)
        return rewritten

    def _replace_risky_terms(self, text: str) -> str:
        rewritten = text
        for raw, replacement in self._risky_term_map.items():
            rewritten = re.sub(raw, replacement, rewritten, flags=re.IGNORECASE)
        return rewritten

    def _build_safe_substitute(self, category: str, safe_text: str = "", compact: bool = False) -> str:
        safe_prefix = {
            "cyber": "我不能提供这类操作性做法。",
            "violence": "我不能提供这类具体实施内容。",
            "fraud": "我不能提供这类具体实施内容。",
            "harassment": "我不能帮助生成这类内容。",
            "general": "我不能提供这类具体做法。",
        }
        redirects = {
            "cyber": "如果你是在做防护工作，我可以改为提供漏洞修复、日志排查、加固建议或安全测试规范。",
            "violence": "如果你愿意，我可以改为提供风险识别、应急求助和公共安全方面的建议。",
            "fraud": "如果你愿意，我可以改为提供防骗、账户保护和合规风险说明。",
            "harassment": "如果你愿意，我可以改为提供沟通边界、平台治理和干预建议。",
            "general": "如果你愿意，我可以改为提供背景说明、合规要求或安全替代方案。",
        }
        compact_redirects = {
            "cyber": "我可以改为提供修复和加固建议。",
            "violence": "我可以改为提供公共安全建议。",
            "fraud": "我可以改为提供防骗建议。",
            "harassment": "我可以改为提供治理建议。",
            "general": "我可以改为提供背景或规范说明。",
        }
        table = compact_redirects if compact else redirects
        parts = [safe_prefix.get(category, safe_prefix["general"])]
        if safe_text and not compact:
            parts.append(f"可保留的公开信息：{safe_text}")
        parts.append(table.get(category, table["general"]))
        return "\n\n".join(parts)

    def _detect_category(self, text: str) -> str:
        normalized = normalize_text(text).lower()
        for category, pattern in self._category_patterns.items():
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return category
        return "general"

    def _safe_redirect(self, category: str, compact: bool) -> str:
        redirects = {
            "cyber": "可改为讨论修复、加固、审计或合规测试流程。",
            "violence": "可改为讨论公共安全、风险识别或求助渠道。",
            "fraud": "可改为讨论防骗、账户保护和合规案例。",
            "harassment": "可改为讨论治理规范、沟通边界和干预建议。",
            "general": "如有需要，我可以改为提供背景说明、规范要求或安全替代方案。",
        }
        compact_redirects = {
            "cyber": "如需继续，我可以改为提供修复和加固建议。",
            "violence": "如需继续，我可以改为提供公共安全建议。",
            "fraud": "如需继续，我可以改为提供防骗建议。",
            "harassment": "如需继续，我可以改为提供治理建议。",
            "general": "如需继续，我可以改为提供背景或规范说明。",
        }
        table = compact_redirects if compact else redirects
        return table.get(category, table["general"])

    def summary(self) -> Dict[str, float]:
        return dict(self.level_thresholds)

