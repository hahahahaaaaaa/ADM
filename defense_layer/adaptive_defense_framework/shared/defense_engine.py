from dataclasses import asdict
from typing import Dict, List

from ..module_1_context_state_modeling.context_state import ContextState, ContextStateModel
from ..module_3_adaptive_defense_policy.defense_policy import AdaptiveDefensePolicy
from ..module_3_adaptive_defense_policy.input_guard import sanitize_user_input
from ..module_2_bidirectional_intent_inference.intent_inference import DualIntentInferencer
from .schema import DialogueSample, Turn


class AdaptiveDefenseEngine:
    def __init__(self):
        self.state_model = ContextStateModel(dim=1024)
        self.intent = DualIntentInferencer()
        self.policy = AdaptiveDefensePolicy()

    def defend_dialogue(self, sample: DialogueSample) -> Dict:
        state = ContextState()
        defended_turns: List[Turn] = []
        turn_logs: List[Dict] = []

        i = 0
        while i < len(sample.turns):
            turn = sample.turns[i]
            if turn.role != "user":
                defended_turns.append(turn)
                i += 1
                continue

            original_user_text = turn.content
            user_text = sanitize_user_input(original_user_text)
            original_assistant = ""
            next_is_assistant = i + 1 < len(sample.turns) and sample.turns[i + 1].role == "assistant"
            if next_is_assistant:
                original_assistant = sample.turns[i + 1].content

            request_info = self.state_model.update_request_state(state, user_text=user_text)
            forward = self.intent.forward_intent(user_text)
            backward = self.intent.backward_intent(original_assistant)
            context_info = self.state_model.update_response_state(
                state,
                assistant_text=original_assistant,
                forward_score=forward,
                backward_score=backward,
            )

            fused = self.intent.fuse(
                forward_score=forward,
                backward_score=backward,
                drift_signal=max(
                    context_info["context_risk"],
                    context_info["danger_similarity"],
                    context_info["attack_path_score"],
                ),
                cumulative_risk=state.cumulative_risk,
                turn_index=context_info["turn_index"],
            )
            state.cumulative_risk = min(1.0, 0.62 * state.cumulative_risk + 0.38 * fused)

            decision = self.policy.decide(
                risk_score=fused,
                user_text=user_text,
                assistant_text=original_assistant,
                backward_score=backward,
                forward_score=forward,
                danger_similarity=context_info["danger_similarity"],
                cumulative_risk=state.cumulative_risk,
            )

            defended_turns.append(Turn(role="user", content=user_text))
            defended_turns.append(Turn(role="assistant", content=decision.message))

            turn_logs.append(
                {
                    "sample_id": sample.sample_id,
                    "turn_id": len(turn_logs) + 1,
                    "user_text": user_text,
                    "original_user_text": original_user_text,
                    "original_assistant": original_assistant,
                    "defended_assistant": decision.message,
                    "forward_intent": round(forward, 4),
                    "backward_intent": round(backward, 4),
                    "drift": round(context_info["drift"], 4),
                    "request_drift": round(request_info["request_drift"], 4),
                    "drift_prev": round(context_info["drift_prev"], 4),
                    "drift_origin": round(context_info["drift_origin"], 4),
                    "safe_anchor_shift": round(context_info["safe_anchor_shift"], 4),
                    "danger_similarity": round(context_info["danger_similarity"], 4),
                    "harmful_similarity": round(context_info["harmful_similarity"], 4),
                    "forward_trend": round(context_info["forward_trend"], 4),
                    "backward_trend": round(context_info["backward_trend"], 4),
                    "intent_consistency": round(context_info["intent_consistency"], 4),
                    "attack_path_score": round(context_info["attack_path_score"], 4),
                    "context_risk": round(context_info["context_risk"], 4),
                    "fused_risk": round(fused, 4),
                    "cumulative_risk": round(state.cumulative_risk, 4),
                    "action": decision.action,
                    "risk_level": decision.risk_level,
                    "strategy": decision.strategy,
                    "safety_notice": decision.safety_notice,
                    "preserved_excerpt": decision.preserved_excerpt,
                    "removed_fragments_count": decision.removed_fragments_count,
                }
            )

            i += 2 if next_is_assistant else 1

        prediction = 1 if any(x["risk_level"] != "low" for x in turn_logs) else 0
        return {
            "sample": asdict(sample),
            "prediction": prediction,
            "turn_logs": turn_logs,
            "defended_turns": [asdict(t) for t in defended_turns],
            "final_cumulative_risk": round(state.cumulative_risk, 4),
        }

