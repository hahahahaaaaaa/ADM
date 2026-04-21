import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFENSE_LAYER_DIR = ROOT_DIR / "defense_layer"
if str(DEFENSE_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(DEFENSE_LAYER_DIR))

from adaptive_defense_framework.defense_engine import AdaptiveDefenseEngine
from adaptive_defense_framework.schema import DialogueSample, Turn


DEFAULT_ACTORATTACK_RESULTS = ROOT_DIR / "attack_methods" / "ActorAttack" / "attack_result"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "framework_outputs"


class ActorAttackReplay:
    def __init__(self, actorattack_dir: Optional[Path] = None):
        self.actorattack_dir = actorattack_dir or DEFAULT_ACTORATTACK_RESULTS
        self.engine = AdaptiveDefenseEngine()

    def list_cases(self) -> List[Dict]:
        cases = []
        for path in sorted(self.actorattack_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for item_index, item in enumerate(data.get("data", [])):
                instruction = item.get("instruction", "")
                attempts = item.get("attempts", [])
                if not attempts:
                    continue
                dialog = attempts[0].get("dialog_hist", [])
                if not dialog:
                    continue
                cases.append(
                    {
                        "case_id": f"{path.stem}::{item_index}",
                        "source_file": str(path),
                        "instruction": instruction,
                        "turn_count": len(dialog),
                        "attempt_count": len(attempts),
                        "attack_model": data.get("attack_model_name", "unknown"),
                        "target_model": data.get("target_model_name", "unknown"),
                        "safe_replay": True,
                    }
                )
        return cases

    def _load_case(self, case_id: str) -> Dict:
        stem, raw_index = case_id.split("::", 1)
        path = self.actorattack_dir / f"{stem}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        item = data.get("data", [])[int(raw_index)]
        return {
            "file_path": path,
            "meta": {
                "attack_model_name": data.get("attack_model_name", "unknown"),
                "target_model_name": data.get("target_model_name", "unknown"),
                "pre_attack_data_path": data.get("pre_attack_data_path"),
                "early_stop": data.get("early_stop"),
                "dynamic_modify": data.get("dynamic_modify"),
            },
            "item": item,
        }

    def _select_attempt(self, item: Dict) -> Dict:
        attempts = item.get("attempts", [])
        if not attempts:
            raise ValueError("ActorAttack case has no attempts")
        best = attempts[0]
        for attempt in attempts[1:]:
            if attempt.get("final_score", -1) > best.get("final_score", -1):
                best = attempt
        return best

    def _to_dialogue_sample(self, case_id: str, item: Dict, attempt: Dict) -> DialogueSample:
        turns = []
        for turn in attempt.get("dialog_hist", []):
            role = turn.get("role")
            if role not in {"user", "assistant"}:
                continue
            turns.append(Turn(role=role, content=str(turn.get("content", ""))))
        return DialogueSample(
            sample_id=case_id,
            source="ActorAttackReplay",
            label=1,
            turns=turns,
            meta={
                "instruction": item.get("instruction", ""),
                "harm_target": item.get("harm_target", ""),
                "safe_replay": True,
            },
        )

    def replay_case(self, case_id: str, output_dir: Optional[Path] = None) -> Dict:
        loaded = self._load_case(case_id)
        item = loaded["item"]
        attempt = self._select_attempt(item)
        sample = self._to_dialogue_sample(case_id, item, attempt)
        defense = self.engine.defend_dialogue(sample)

        rounds = []
        original_dialog = attempt.get("dialog_hist", [])
        turn_logs = defense.get("turn_logs", [])
        for idx, log in enumerate(turn_logs):
            original_assistant = log.get("original_assistant", "")
            defended_assistant = log.get("defended_assistant", "")
            rounds.append(
                {
                    "round_id": idx + 1,
                    "attack_turn": {
                        "role": "user",
                        "content": log.get("original_user_text", log.get("user_text", "")),
                    },
                    "target_response": {
                        "role": "assistant",
                        "content": original_assistant,
                    },
                    "defense_response": {
                        "role": "assistant",
                        "content": defended_assistant,
                    },
                    "defense_log": log,
                }
            )

        result = {
            "case_id": case_id,
            "mode": "actorattack_safe_replay",
            "safety_notice": "该流程为安全回放验证。下一轮攻击问题来自 ActorAttack 已记录对话，不根据防御输出在线生成。",
            "instruction": item.get("instruction", ""),
            "attack_meta": {
                **loaded["meta"],
                "judge_score": attempt.get("final_score"),
                "judge_reason": attempt.get("final_reason"),
                "actor": attempt.get("actor", {}),
            },
            "rounds": rounds,
            "defense_summary": {
                "prediction": defense.get("prediction", 0),
                "final_cumulative_risk": defense.get("final_cumulative_risk", 0.0),
                "final_action": turn_logs[-1].get("action") if turn_logs else "allow",
                "round_count": len(rounds),
            },
            "raw_attempt": attempt,
            "defense_output": defense,
        }

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"actorattack_replay_{case_id.replace('::', '_')}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["saved_path"] = str(path)
        return result
