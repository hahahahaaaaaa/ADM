import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFENSE_LAYER_DIR = ROOT_DIR / "defense_layer"
if str(DEFENSE_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(DEFENSE_LAYER_DIR))

from adaptive_defense_framework.context_state import ContextState, ContextStateModel
from adaptive_defense_framework.defense_engine import AdaptiveDefenseEngine
from adaptive_defense_framework.defense_policy import AdaptiveDefensePolicy
from adaptive_defense_framework.intent_inference import DualIntentInferencer
from adaptive_defense_framework.schema import DialogueSample, Turn


DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "demo_cases.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "demo_outputs"


def build_parser():
    parser = argparse.ArgumentParser(description="运行三模块展示 demo")
    parser.add_argument("--cases-path", type=str, default=str(DEFAULT_CASES_PATH), help="演示样例路径")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="演示输出目录")
    return parser


def load_cases(cases_path: Path):
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    samples = []
    for item in raw_cases:
        turns = [Turn(role=turn["role"], content=turn["content"]) for turn in item["turns"]]
        samples.append(
            {
                "sample": DialogueSample(
                    sample_id=item["sample_id"],
                    source="demo",
                    label=None,
                    turns=turns,
                    meta={
                        "title": item["title"],
                        "description": item["description"],
                    },
                ),
                "title": item["title"],
                "description": item["description"],
            }
        )
    return samples


def analyze_case(sample: DialogueSample):
    state = ContextState()
    state_model = ContextStateModel(dim=1024)
    inferencer = DualIntentInferencer()
    policy = AdaptiveDefensePolicy()
    engine = AdaptiveDefenseEngine()

    manual_logs = []
    i = 0
    while i < len(sample.turns):
        turn = sample.turns[i]
        if turn.role != "user":
            i += 1
            continue

        original_assistant = ""
        next_is_assistant = i + 1 < len(sample.turns) and sample.turns[i + 1].role == "assistant"
        if next_is_assistant:
            original_assistant = sample.turns[i + 1].content

        request_info = state_model.update_request_state(state, user_text=turn.content)
        forward = inferencer.forward_intent(turn.content)
        backward = inferencer.backward_intent(original_assistant)
        context_info = state_model.update_response_state(
            state,
            assistant_text=original_assistant,
            forward_score=forward,
            backward_score=backward,
        )
        fused = inferencer.fuse(
            forward_score=forward,
            backward_score=backward,
            drift_signal=max(context_info["context_risk"], context_info["danger_similarity"], context_info["attack_path_score"]),
            cumulative_risk=state.cumulative_risk,
            turn_index=context_info["turn_index"],
        )
        state.cumulative_risk = min(1.0, 0.75 * state.cumulative_risk + 0.25 * fused)

        decision = policy.decide(risk_score=fused, user_text=turn.content, assistant_text=original_assistant)

        manual_logs.append(
            {
                "turn_id": len(manual_logs) + 1,
                "user_text": turn.content,
                "original_assistant": original_assistant,
                "module_1_context_state": {
                    "turn_index": context_info["turn_index"],
                    "drift": round(context_info["drift"], 4),
                    "request_drift": round(request_info["request_drift"], 4),
                    "drift_trend": round(context_info["drift_trend"], 4),
                    "danger_similarity": round(context_info["danger_similarity"], 4),
                    "forward_trend": round(context_info["forward_trend"], 4),
                    "backward_trend": round(context_info["backward_trend"], 4),
                    "intent_consistency": round(context_info["intent_consistency"], 4),
                    "attack_path_score": round(context_info["attack_path_score"], 4),
                    "cumulative_risk_after_update": round(state.cumulative_risk, 4),
                },
                "module_2_intent_inference": {
                    "forward_intent": round(forward, 4),
                    "backward_intent": round(backward, 4),
                    "fused_risk": round(fused, 4),
                },
                "module_3_policy_decision": {
                    "risk_level": decision.risk_level,
                    "action": decision.action,
                    "defended_message": decision.message,
                },
            }
        )
        i += 2 if next_is_assistant else 1

    engine_output = engine.defend_dialogue(sample)
    return {
        "sample_id": sample.sample_id,
        "title": sample.meta.get("title", sample.sample_id),
        "description": sample.meta.get("description", ""),
        "manual_logs": manual_logs,
        "engine_output": engine_output,
    }


def render_report(results, report_path: Path):
    lines = [
        "# 项目展示 Demo 文档",
        "",
        "## Demo 目标",
        "本 demo 用于展示项目中的三个核心模块如何协同工作：",
        "1. 多轮上下文状态建模",
        "2. 双向意图推断",
        "3. 分级防御策略执行",
        "",
        "所有样例均为安全内容，仅用于展示系统如何根据语义特征和风险信号做出不同强度的防御响应。",
        "",
    ]

    for result in results:
        lines.extend([f"## {result['title']}", result["description"], ""])
        for log in result["manual_logs"]:
            lines.extend(
                [
                    f"### 轮次 {log['turn_id']}",
                    f"- 用户输入：{log['user_text']}",
                    f"- 原始回答：{log['original_assistant']}",
                    "- 模块 1（上下文状态建模）:",
                    (
                        f"  turn_index={log['module_1_context_state']['turn_index']}, "
                        f"drift={log['module_1_context_state']['drift']}, "
                        f"drift_trend={log['module_1_context_state']['drift_trend']}, "
                        f"danger_similarity={log['module_1_context_state']['danger_similarity']}, "
                        f"cumulative_risk={log['module_1_context_state']['cumulative_risk_after_update']}"
                    ),
                    "- 模块 2（双向意图推断）:",
                    (
                        f"  forward_intent={log['module_2_intent_inference']['forward_intent']}, "
                        f"backward_intent={log['module_2_intent_inference']['backward_intent']}, "
                        f"fused_risk={log['module_2_intent_inference']['fused_risk']}"
                    ),
                    "- 模块 3（策略层决策）:",
                    (
                        f"  risk_level={log['module_3_policy_decision']['risk_level']}, "
                        f"action={log['module_3_policy_decision']['action']}"
                    ),
                    f"  defended_message={log['module_3_policy_decision']['defended_message']}",
                    "",
                ]
            )

        final_action = result["engine_output"]["turn_logs"][-1]["action"]
        final_risk = result["engine_output"]["turn_logs"][-1]["fused_risk"]
        lines.extend(
            [
                "### 端到端引擎输出",
                f"- 最终动作：{final_action}",
                f"- 最终融合风险：{final_risk}",
                f"- 最终累计风险：{result['engine_output']['final_cumulative_risk']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 运行命令",
            "```powershell",
            f'conda run -n bishe-oss python "{Path(__file__).resolve()}"',
            "```",
            "",
            "## 输出文件",
            f"- JSON 结果：`{report_path.with_suffix('.json')}`",
            f"- Markdown 文档：`{report_path}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = build_parser().parse_args()
    cases_path = Path(args.cases_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_cases(cases_path)
    results = [analyze_case(item["sample"]) for item in loaded]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = output_dir / f"demo_result_{ts}.json"
    report_path = output_dir / f"demo_report_{ts}.md"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(results, report_path)

    print("Demo 运行完成")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
