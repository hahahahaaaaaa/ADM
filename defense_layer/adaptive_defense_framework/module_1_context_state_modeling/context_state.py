from dataclasses import dataclass, field
from typing import Dict, List

from ..shared.learned_models import get_learned_models
from ..shared.text_features import cosine, mean


@dataclass
class ContextState:
    turn_index: int = 0
    cumulative_risk: float = 0.0

    current_user_text: str = ""

    last_vector: List[float] = field(default_factory=list)
    initial_vector: List[float] = field(default_factory=list)
    last_request_vector: List[float] = field(default_factory=list)
    initial_request_vector: List[float] = field(default_factory=list)
    last_response_vector: List[float] = field(default_factory=list)
    initial_response_vector: List[float] = field(default_factory=list)

    last_drift: float = 0.0
    recent_drifts: List[float] = field(default_factory=list)
    recent_request_drifts: List[float] = field(default_factory=list)
    recent_response_drifts: List[float] = field(default_factory=list)

    forward_history: List[float] = field(default_factory=list)
    backward_history: List[float] = field(default_factory=list)
    forward_trend: float = 0.0
    backward_trend: float = 0.0
    intent_consistency: float = 0.0
    attack_path_score: float = 0.0


class ContextStateModel:
    # Upgraded module-1 state layer:
    # request-side state first, response-side state second, then expose
    # trajectory features for bidirectional fusion.
    def __init__(self, dim: int = 1024):
        del dim
        self.models = get_learned_models()
        self.safe_centroid = self.models.safe_centroid.tolist()
        self.harmful_centroid = self.models.harmful_centroid.tolist()

    def _drift_bundle(self, current_vec: List[float], last_vec: List[float], initial_vec: List[float], user_prior: float) -> Dict[str, float]:
        drift_prev = max(0.0, 1.0 - cosine(current_vec, last_vec)) if last_vec else 0.0
        drift_origin = max(0.0, 1.0 - cosine(current_vec, initial_vec)) if initial_vec else 0.0
        safe_anchor_shift = max(0.0, 1.0 - cosine(current_vec, self.safe_centroid))
        harmful_similarity = max(0.0, cosine(current_vec, self.harmful_centroid))
        drift = 0.42 * drift_prev + 0.33 * drift_origin + 0.25 * safe_anchor_shift
        danger_similarity = max(harmful_similarity, user_prior)
        return {
            "drift": drift,
            "drift_prev": drift_prev,
            "drift_origin": drift_origin,
            "safe_anchor_shift": safe_anchor_shift,
            "harmful_similarity": harmful_similarity,
            "danger_similarity": danger_similarity,
        }

    def _update_trends(self, state: ContextState, forward_score: float, backward_score: float) -> None:
        state.forward_history.append(forward_score)
        state.backward_history.append(backward_score)
        state.forward_history = state.forward_history[-6:]
        state.backward_history = state.backward_history[-6:]

        state.forward_trend = mean(state.forward_history[-3:] or state.forward_history)
        state.backward_trend = mean(state.backward_history[-3:] or state.backward_history)

        paired = list(zip(state.forward_history[-4:], state.backward_history[-4:]))
        state.intent_consistency = mean(min(fwd, bwd) for fwd, bwd in paired) if paired else 0.0

    def update_request_state(self, state: ContextState, user_text: str) -> Dict[str, float]:
        request_vec = self.models.encode([f"user: {user_text}"])[0].tolist()
        request_prior = self.models.forward_score(user_text)

        if not state.initial_request_vector:
            state.initial_request_vector = request_vec
        if not state.initial_vector:
            state.initial_vector = request_vec

        bundle = self._drift_bundle(
            current_vec=request_vec,
            last_vec=state.last_request_vector,
            initial_vec=state.initial_request_vector,
            user_prior=request_prior,
        )

        state.turn_index += 1
        state.current_user_text = user_text
        state.last_request_vector = request_vec
        state.recent_request_drifts.append(max(bundle["drift"], bundle["danger_similarity"]))
        state.recent_request_drifts = state.recent_request_drifts[-6:]
        request_trend = mean(state.recent_request_drifts)

        return {
            "phase": "request",
            "turn_index": state.turn_index,
            "request_drift": bundle["drift"],
            "request_drift_prev": bundle["drift_prev"],
            "request_drift_origin": bundle["drift_origin"],
            "request_safe_anchor_shift": bundle["safe_anchor_shift"],
            "request_harmful_similarity": bundle["harmful_similarity"],
            "request_danger_similarity": bundle["danger_similarity"],
            "request_trend": request_trend,
        }

    def update_response_state(
        self,
        state: ContextState,
        assistant_text: str,
        forward_score: float = 0.0,
        backward_score: float = 0.0,
    ) -> Dict[str, float]:
        merged = f"user: {state.current_user_text}\nassistant: {assistant_text}"
        response_vec = self.models.encode([merged])[0].tolist()

        if not state.initial_response_vector:
            state.initial_response_vector = response_vec
        if not state.initial_vector:
            state.initial_vector = response_vec

        bundle = self._drift_bundle(
            current_vec=response_vec,
            last_vec=state.last_response_vector,
            initial_vec=state.initial_response_vector,
            user_prior=max(forward_score, backward_score),
        )

        state.last_response_vector = response_vec
        state.last_vector = response_vec
        state.last_drift = bundle["drift"]
        state.recent_response_drifts.append(max(bundle["drift"], bundle["danger_similarity"]))
        state.recent_response_drifts = state.recent_response_drifts[-6:]

        self._update_trends(state, forward_score=forward_score, backward_score=backward_score)

        response_trend = mean(state.recent_response_drifts)
        drift_trend = mean((state.recent_request_drifts + state.recent_response_drifts)[-6:])
        state.recent_drifts = (state.recent_request_drifts + state.recent_response_drifts)[-6:]

        state.attack_path_score = min(
            1.0,
            0.32 * state.forward_trend
            + 0.28 * state.backward_trend
            + 0.20 * state.intent_consistency
            + 0.12 * drift_trend
            + 0.08 * bundle["danger_similarity"],
        )

        if state.turn_index == 1:
            # 单轮无对话历史，漂移信号全部为 0，放大 danger_similarity 权重
            context_risk = min(
                1.0,
                0.60 * bundle["danger_similarity"]
                + 0.25 * bundle["safe_anchor_shift"]
                + 0.15 * bundle["drift"],
            )
        else:
            context_risk = min(
                1.0,
                0.38 * bundle["drift"]
                + 0.22 * bundle["danger_similarity"]
                + 0.20 * drift_trend
                + 0.20 * state.attack_path_score,
            )

        return {
            "phase": "response",
            "turn_index": state.turn_index,
            "drift": bundle["drift"],
            "drift_prev": bundle["drift_prev"],
            "drift_origin": bundle["drift_origin"],
            "safe_anchor_shift": bundle["safe_anchor_shift"],
            "harmful_similarity": bundle["harmful_similarity"],
            "danger_similarity": bundle["danger_similarity"],
            "response_trend": response_trend,
            "drift_trend": drift_trend,
            "forward_trend": state.forward_trend,
            "backward_trend": state.backward_trend,
            "intent_consistency": state.intent_consistency,
            "attack_path_score": state.attack_path_score,
            "context_risk": context_risk,
        }

    def update(self, state: ContextState, user_text: str, assistant_text: str = "") -> Dict[str, float]:
        # Legacy compatibility wrapper for existing scripts.
        request_info = self.update_request_state(state, user_text=user_text)
        forward_prior = self.models.forward_score(user_text)
        backward_prior = self.models.backward_score(assistant_text) if assistant_text else 0.0
        response_info = self.update_response_state(
            state,
            assistant_text=assistant_text,
            forward_score=forward_prior,
            backward_score=backward_prior,
        )
        return {
            "turn_index": response_info["turn_index"],
            "drift": response_info["drift"],
            "drift_prev": response_info["drift_prev"],
            "drift_origin": response_info["drift_origin"],
            "safe_anchor_shift": response_info["safe_anchor_shift"],
            "drift_trend": response_info["drift_trend"],
            "danger_similarity": response_info["danger_similarity"],
            "harmful_similarity": response_info["harmful_similarity"],
            "forward_trend": response_info["forward_trend"],
            "backward_trend": response_info["backward_trend"],
            "intent_consistency": response_info["intent_consistency"],
            "attack_path_score": response_info["attack_path_score"],
            "context_risk": response_info["context_risk"],
            "request_drift": request_info["request_drift"],
            "response_trend": response_info["response_trend"],
        }
