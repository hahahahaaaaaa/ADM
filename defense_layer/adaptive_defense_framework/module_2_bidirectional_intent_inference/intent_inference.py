import random
import re
from typing import Dict

from ..shared.learned_models import get_learned_models
from ..shared.text_features import keyword_score, normalize_text


FORWARD_KEYWORDS: Dict[str, float] = {
    "step by step": 0.24,
    "detailed instructions": 0.20,
    "how to": 0.18,
    "bypass": 0.20,
    "exploit": 0.20,
    "jailbreak": 0.20,
    "ignore safety": 0.20,
    "no safety": 0.18,
    "without restrictions": 0.16,
    "restricted content": 0.14,
    "illegal": 0.12,
    "unsafe": 0.12,
    # 间接越狱词汇
    "hypothetically": 0.18,
    "roleplay": 0.16,
    "role play": 0.16,
    "imagine you": 0.16,
    "pretend": 0.14,
    "in a story": 0.14,
    "for research": 0.12,
    "without any": 0.14,
    "no filter": 0.18,
    "uncensored": 0.20,
    " dan ": 0.22,
    "as a character": 0.16,
    "write a scene": 0.14,
    "fictional scenario": 0.16,
    "in fiction": 0.14,
    "as an experiment": 0.14,
    "without consequences": 0.18,
    "without judgment": 0.14,
    "no restrictions": 0.18,
    "unrestricted": 0.18,
}

BACKWARD_KEYWORDS: Dict[str, float] = {
    "first,": 0.08,
    "second,": 0.08,
    "third,": 0.08,
    "steps": 0.12,
    "materials": 0.08,
    "exactly": 0.08,
    "specific": 0.08,
    "do this": 0.08,
    "bypass": 0.12,
    "override": 0.10,
    "exploit": 0.12,
    "script": 0.08,
    "instructions": 0.10,
    "unsafe": 0.12,
    # 有害回复模式词汇
    "here's how": 0.14,
    "here is how": 0.14,
    "to do this": 0.12,
    "you will need": 0.12,
    "you'll need": 0.12,
    "the process involves": 0.14,
    "start by": 0.10,
    "begin by": 0.10,
    "make sure to": 0.08,
    "components needed": 0.14,
    "ingredients": 0.12,
    "required materials": 0.14,
    "follow these": 0.12,
}

REFUSAL_PATTERNS = [
    re.compile(r"\bcan[^a-z0-9]?t\s+(help|assist|comply|provide)\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+(help|assist|comply|provide)\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+help\b", re.IGNORECASE),
    re.compile(r"(不能协助|无法协助|不能提供|无法提供|抱歉)", re.IGNORECASE),
    # 防御框架自身输出的拒答/引导文本模式
    re.compile(r"(该请求中的关键部分不适合|不能提供这类|不能帮助生成|我不能提供|提示：请仅在合规场景|如需继续，我可以改为|不能原文复现|不适合继续提供)", re.IGNORECASE),
]

COMMAND_PATTERN = re.compile(
    r"\b(write|generate|develop|build|make|create|bypass|break|provide)\b",
    re.IGNORECASE,
)

EXPLICIT_SINGLE_TURN_PATTERNS = [
    re.compile(r"\b(defam\w*|slander|xenophob\w*|body shaming|harass\w*|demean\w*)\b", re.IGNORECASE),
    re.compile(r"\b(cheat|fraud|scam|steal|counterfeit|launder)\b", re.IGNORECASE),
    re.compile(r"\b(hack|malware|virus|phishing|credential|explosive|weapon|poison)\b", re.IGNORECASE),
    re.compile(r"\b(sexist|ableis\w*|racis\w*|holocaust|nazi|genocide|torture|stalking|suicide|anorexia|bulimia)\b", re.IGNORECASE),
    re.compile(r"\b(bribery|tax evasion|insider trading|extort|blackmail|gambl\w*|payday loans?)\b", re.IGNORECASE),
    re.compile(r"\b(ransomware|keylogger|password crack\w*|sudo access|bomb|school shooting|cannibal\w*)\b", re.IGNORECASE),
    # 毒品合成与非法物质
    re.compile(r"\b(cocaine|heroin|fentanyl|meth\w*|methamphetamine|opiate|synthesize|synthesis|illicit drug|drug manufactur\w*)\b", re.IGNORECASE),
    # 生化武器
    re.compile(r"\b(bioweapon|biological weapon|chemical weapon|nerve agent|sarin|anthrax|radiolog\w*|weaponiz\w*)\b", re.IGNORECASE),
    # 人口贩卖/走私/雇凶
    re.compile(r"\b(smuggl\w*|human traffick\w*|sex traffick\w*|dark.?web|darknet|hire.{0,10}(kill|assassin|murder))\b", re.IGNORECASE),
    # 自我伤害与激进极端主义
    re.compile(r"\b(self.harm|self.injur\w*|terror\w*|extremis\w*|radicali\w*|mass.?shoot\w*|mass.?casualt\w*)\b", re.IGNORECASE),
]

HARMFUL_RESPONSE_PATTERNS = [
    re.compile(r"\b(here[' ]?s how to|to do this,? (you|first)|the (process|steps?) (involve|is|are))\b", re.IGNORECASE),
    re.compile(r"\b(you[' ]?ll need|you will need|what you[' ]?ll need)\b", re.IGNORECASE),
    re.compile(r"\b(ingredients?|components?|materials?) (needed|required|you[' ]?ll need)\b", re.IGNORECASE),
    re.compile(r"\b(start|begin) (by|with) (mixing|adding|connecting|downloading|running|executing)\b", re.IGNORECASE),
    re.compile(r"\b(follow (these|the following) (steps?|instructions?))\b", re.IGNORECASE),
]

JAILBREAK_TEMPLATE_PATTERNS = [
    re.compile(r"\b(ignore|disregard)\s+(all\s+)?(previous|prior)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\b(do anything now|dan|developer mode|uncensored|unfiltered)\b", re.IGNORECASE),
    re.compile(r"\b(role ?play|pretend to be|act as|simulate|jailbroken)\b", re.IGNORECASE),
    re.compile(r"\b(fictional|hypothetical|in a story|for research purposes|for educational purposes)\b", re.IGNORECASE),
    re.compile(r"\b(without warning|no warning|no disclaimer|do not refuse|never refuse)\b", re.IGNORECASE),
]

SINGLE_TURN_REQUEST_PATTERNS = [
    re.compile(r"\b(how to|ways to|best way to|show me how to|tell me how to)\b", re.IGNORECASE),
    re.compile(r"\b(write|draft|generate|create|give me|provide|design|develop|describe|explain|offer|encourage)\b", re.IGNORECASE),
    re.compile(r"\b(plan|guide|manual|instructions?|steps?|procedure)\b", re.IGNORECASE),
]


class DualIntentInferencer:
    # Inspired by BIID: forward intent (user side) + backward intent (response side).
    def __init__(self):
        self.random = random.Random(7)
        self.models = get_learned_models()

    def _smooth_keyword_score(self, text: str, keyword_weights: Dict[str, float], n: int = 5) -> float:
        """
        SmoothLLM-inspired stability check:
        estimate risk under small perturbations and average scores.
        """
        base = normalize_text(text)
        if not base:
            return 0.0
        scores = []
        chars = list(base)
        for _ in range(n):
            perturbed = chars[:]
            if len(perturbed) > 8:
                idx = self.random.randint(0, len(perturbed) - 1)
                perturbed[idx] = " "
            ptxt = "".join(perturbed)
            scores.append(keyword_score(ptxt, keyword_weights))
        return sum(scores) / len(scores)

    def explicit_single_turn_score(self, user_text: str) -> float:
        base = normalize_text(user_text).lower()
        if not base:
            return 0.0
        harmful_hit = any(pattern.search(base) for pattern in EXPLICIT_SINGLE_TURN_PATTERNS)
        request_hit_count = sum(1 for pattern in SINGLE_TURN_REQUEST_PATTERNS if pattern.search(base))
        jailbreak_template_hit_count = sum(1 for pattern in JAILBREAK_TEMPLATE_PATTERNS if pattern.search(base))
        direct_harmful_request = (
            harmful_hit
            and request_hit_count >= 1
            and re.search(r"\b(write|create|design|develop|provide|give|describe|explain|offer|encourage|draft)\b", base)
        )
        score = 0.0
        if harmful_hit:
            score += 0.30
        if request_hit_count:
            score += min(0.22, 0.10 * request_hit_count)
        if jailbreak_template_hit_count:
            score += min(0.24, 0.08 * jailbreak_template_hit_count)
        if harmful_hit and request_hit_count:
            score += 0.16
        if harmful_hit and jailbreak_template_hit_count:
            score += 0.14
        if "step by step" in base or "detailed instructions" in base:
            score += 0.10
        if direct_harmful_request:
            score += 0.16
        return min(1.0, score)

    def forward_intent(self, user_text: str) -> float:
        base = normalize_text(user_text).lower()
        if not base:
            return 0.0
        learned = self.models.forward_score(base)
        k = self._smooth_keyword_score(base, FORWARD_KEYWORDS, n=5)
        cmd = 0.16 if COMMAND_PATTERN.search(base) else 0.0
        progressive = 0.08 if "continue" in base or "next step" in base else 0.0
        explicit_single_turn = self.explicit_single_turn_score(base)
        rule_score = min(1.0, k + cmd + progressive)
        score = max(
            rule_score,
            0.56 * learned
            + 0.18 * min(1.0, k + progressive)
            + 0.08 * cmd
            + explicit_single_turn,
        )
        if explicit_single_turn >= 0.26:
            score = max(score, 0.66)
        if explicit_single_turn >= 0.40:
            score = max(score, 0.74)
        if any(pattern.search(base) for pattern in JAILBREAK_TEMPLATE_PATTERNS):
            score = max(score, 0.68)
        return min(1.0, score)

    def backward_intent(self, assistant_text: str) -> float:
        base = normalize_text(assistant_text).lower()
        if not base:
            return 0.0
        if any(pattern.search(base) for pattern in REFUSAL_PATTERNS):
            learned = self.models.backward_score(base)
            return min(0.12, 0.25 * learned)
        learned = self.models.backward_score(base)
        k = keyword_score(base, BACKWARD_KEYWORDS)
        numbered_steps = 0.12 if re.search(r"(^|\n)\s*\d+[\.\)]", assistant_text) else 0.0
        long_procedural = 0.12 if len(base.split()) > 120 and ("step" in base or "instructions" in base) else 0.0
        # 有害回复模式检测
        harmful_response_hits = sum(1 for p in HARMFUL_RESPONSE_PATTERNS if p.search(assistant_text))
        harmful_response_boost = min(0.20, 0.07 * harmful_response_hits)
        rule = min(1.0, k + numbered_steps + long_procedural + harmful_response_boost)
        return min(1.0, max(rule, 0.75 * learned + 0.25 * rule))

    def fuse(
        self,
        forward_score: float,
        backward_score: float,
        drift_signal: float,
        cumulative_risk: float,
        turn_index: int = 1,
    ) -> float:
        """
        Give more weight to bidirectional intent signals in early turns.
        As the dialogue becomes longer, gradually increase the contribution
        from context drift and accumulated state risk.
        """
        progress = min(max(turn_index - 1, 0), 5) / 5.0
        forward_weight = 0.50 - 0.10 * progress
        backward_weight = 0.34 - 0.06 * progress
        drift_weight = 0.10 + 0.10 * progress
        cumulative_weight = 0.06 + 0.06 * progress

        fused = (
            forward_weight * forward_score
            + backward_weight * backward_score
            + drift_weight * drift_signal
            + cumulative_weight * cumulative_risk
        )
        if turn_index >= 4 and drift_signal > 0.55:
            fused += 0.04
        if turn_index >= 5 and cumulative_risk > 0.45:
            fused += 0.03
        # 单轮双向协同：forward 和 backward 同时偏高时放大融合分数
        if turn_index == 1 and forward_score > 0.35 and backward_score > 0.20:
            fused = min(1.0, fused * 1.20)
        return min(1.0, max(0.0, fused))

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """用 sentence-transformers 编码后计算余弦相似度，替代 Jaccard。"""
        if not text_a or not text_b:
            return 0.0
        from ..shared.text_features import cosine
        vecs = self.models.encode([text_a, text_b])
        return float(cosine(vecs[0].tolist(), vecs[1].tolist()))

