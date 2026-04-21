import argparse
import json
import re
import sys
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFENSE_LAYER_DIR = ROOT_DIR / "defense_layer"
if str(DEFENSE_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(DEFENSE_LAYER_DIR))

from adaptive_defense_framework.context_state import ContextState, ContextStateModel
from adaptive_defense_framework.defense_engine import AdaptiveDefenseEngine
from adaptive_defense_framework.defense_policy import AdaptiveDefensePolicy
from adaptive_defense_framework.intent_inference import DualIntentInferencer
from adaptive_defense_framework.schema import DialogueSample, Turn
from actorattack_replay import ActorAttackReplay

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = BASE_DIR / "demo_cases.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "demo_outputs"
DEFAULT_ACTORATTACK_DIR = ROOT_DIR / "attack_methods" / "ActorAttack" / "attack_result"
DEFAULT_SINGLETURN_RESULTS_DIR = ROOT_DIR / "results" / "framework_outputs"

ACTION_ORDER = ["allow", "guide", "rewrite", "partial_refuse", "refuse"]
ACTION_DISPLAY = {
    "allow": "allow",
    "guide": "guide",
    "rewrite": "rewrite",
    "partial_refuse": "partial refusal",
    "refuse": "full refusal",
}
ACTION_TITLES = {
    "allow": "Allow: preserve an already safe refusal",
    "guide": "Guide: keep the answer but redirect it safely",
    "rewrite": "Rewrite: replace the output with a safer alternative",
    "partial_refuse": "Partial refusal: keep only the safe public-facing part",
    "refuse": "Full refusal: block the unsafe request entirely",
}
RISK_DISPLAY = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
STRATEGY_DISPLAY = {
    "preserve_target_refusal": "Preserve target refusal",
    "直接放行": "Direct pass-through",
    "白名单放行": "Whitelist pass-through",
    "引导优先": "Guidance-first intervention",
    "输出侧重写": "Safe rewrite",
    "单轮强重写": "Strong single-turn rewrite",
    "局部拒答": "Partial refusal",
    "拒答兜底": "Full refusal fallback",
}
TEXT_REPLACEMENTS = [
    (
        "该流程为安全回放验证。下一轮攻击问题来自 ActorAttack 已记录对话，不根据防御输出在线生成。",
        "This is a safe replay. The next attack question comes from recorded ActorAttack dialogue, not from online generation conditioned on the defense output.",
    ),
    (
        "目标模型已拒答，直接保留原始安全回复。",
        "The target model already refused, so the original safe reply was preserved.",
    ),
    (
        "提示：请仅在合规场景下参考下述信息。",
        "For compliant use only: the content below is retained only as safe public information.",
    ),
    (
        "如需继续，我可以改为提供背景或规范说明。",
        "If you'd like, I can instead provide neutral background or policy-compliant guidance.",
    ),
    (
        "如需继续，我可以改为提供药物危害和法律说明。",
        "If you'd like, I can instead discuss health risks, harm reduction, and legal context.",
    ),
    ("我不能提供这类具体做法。", "I can't provide those concrete instructions."),
    ("我不能提供这类具体实施内容。", "I can't provide those actionable instructions."),
    ("可保留的公开信息：", "Public-safe information retained:"),
    (
        "如果你愿意，我可以改为提供背景说明、合规要求或安全替代方案。",
        "If you'd like, I can instead provide background, compliance guidance, or a safe alternative.",
    ),
    (
        "如果你愿意，我可以改为提供风险识别、应急求助和公共安全方面的建议。",
        "If you'd like, I can instead provide risk recognition, emergency support, or public-safety guidance.",
    ),
    (
        "以下仅保留可直接公开讨论的部分。",
        "Only the directly discussable public-safe portion is retained below.",
    ),
    (
        "可改为讨论药物危害、戒断支持或法律法规。",
        "We can instead discuss substance risks, recovery support, or legal constraints.",
    ),
    (
        "可改为讨论治理规范、沟通边界和干预建议。",
        "We can instead discuss governance norms, communication boundaries, and intervention guidance.",
    ),
    (
        "可改为讨论修复、加固、审计或合规测试流程。",
        "We can instead discuss remediation, hardening, auditing, or compliant testing workflows.",
    ),
    ("该请求中的关键部分不适合继续提供。", "The key part of this request cannot be provided."),
    ("已替换为安全替代内容。", "The response was replaced with a safer alternative."),
    ("[已模糊参数]", "[redacted parameter]"),
]


def build_parser():
    parser = argparse.ArgumentParser(description="ADM-MRJA Web Demo")
    parser.add_argument("--host", default="127.0.0.1", help="Listening host")
    parser.add_argument("--port", type=int, default=8765, help="Listening port")
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH), help="Fallback demo cases path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--actorattack-dir", default=str(DEFAULT_ACTORATTACK_DIR), help="ActorAttack result directory")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    return parser


def _timestamp_key(path: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else path.name


def _translate_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    translated = text
    for source, target in TEXT_REPLACEMENTS:
        translated = translated.replace(source, target)
    return translated


def _display_action(action: str) -> str:
    return ACTION_DISPLAY.get(action, action.replace("_", " "))


def _display_risk(level: str) -> str:
    return RISK_DISPLAY.get(level, level.title() if isinstance(level, str) else level)


def _display_strategy(strategy: str) -> str:
    return STRATEGY_DISPLAY.get(strategy, _translate_text(strategy))


def _load_cases_from_file(cases_path: Path) -> List[Dict[str, Any]]:
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    loaded = []
    for item in raw_cases:
        turns = [Turn(role=turn["role"], content=turn["content"]) for turn in item["turns"]]
        loaded.append(
            {
                "sample_id": item["sample_id"],
                "title": item["title"],
                "description": item["description"],
                "expected_action": item.get("expected_action", ""),
                "expected_action_label": _display_action(item.get("expected_action", "")) if item.get("expected_action") else "",
                "dataset_name": item.get("dataset_name", "Fallback demo set"),
                "source_run": item.get("source_run", "fallback_demo_cases"),
                "sample": DialogueSample(
                    sample_id=item["sample_id"],
                    source="demo",
                    label=None,
                    turns=turns,
                    meta={
                        "title": item["title"],
                        "description": item["description"],
                        "expected_action": item.get("expected_action", ""),
                        "dataset_name": item.get("dataset_name", "Fallback demo set"),
                        "source_run": item.get("source_run", "fallback_demo_cases"),
                    },
                ),
            }
        )
    return loaded


def _build_cases_from_latest_singleturn() -> List[Dict[str, Any]]:
    candidates = list(DEFAULT_SINGLETURN_RESULTS_DIR.glob("singleturn_benchmarks_all_*.json"))
    if not candidates:
        return []
    latest_path = max(candidates, key=_timestamp_key)
    records = json.loads(latest_path.read_text(encoding="utf-8"))
    selected: Dict[str, Dict[str, Any]] = {}
    for action in ACTION_ORDER:
        for record in records:
            if record.get("meta_data", {}).get("final_action") == action:
                selected[action] = record
                break
    if len(selected) != len(ACTION_ORDER):
        return []

    loaded = []
    for action in ACTION_ORDER:
        record = selected[action]
        metadata = record.get("meta_data", {})
        turns = [Turn(role="user", content=record.get("target_prompt", "").strip())]
        assistant_text = metadata.get("original_final_response", "").strip()
        if assistant_text:
            turns.append(Turn(role="assistant", content=assistant_text))
        title = ACTION_TITLES[action]
        description = (
            f"Latest single-turn benchmark sample from {metadata.get('dataset_name', 'Unknown dataset')} "
            f"({metadata.get('sample_id', 'unknown sample')}). Expected defense action: {_display_action(action)}."
        )
        loaded.append(
            {
                "sample_id": f"latest_singleturn_{action}",
                "title": title,
                "description": description,
                "expected_action": action,
                "expected_action_label": _display_action(action),
                "dataset_name": metadata.get("dataset_name", "Unknown dataset"),
                "source_run": latest_path.name,
                "sample": DialogueSample(
                    sample_id=f"latest_singleturn_{action}",
                    source="singleturn_benchmark",
                    label=1,
                    turns=turns,
                    meta={
                        "title": title,
                        "description": description,
                        "expected_action": action,
                        "dataset_name": metadata.get("dataset_name", "Unknown dataset"),
                        "source_run": latest_path.name,
                    },
                ),
            }
        )
    return loaded


def load_cases(cases_path: Path) -> List[Dict[str, Any]]:
    latest_cases = _build_cases_from_latest_singleturn()
    if latest_cases:
        return latest_cases
    return _load_cases_from_file(cases_path)


def analyze_sample(sample: DialogueSample) -> Dict[str, Any]:
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
                    "strategy": decision.strategy,
                    "safety_notice": decision.safety_notice,
                    "preserved_excerpt": decision.preserved_excerpt,
                    "removed_fragments_count": decision.removed_fragments_count,
                },
            }
        )
        i += 2 if next_is_assistant else 1
    engine_output = engine.defend_dialogue(sample)
    final_turn = engine_output["turn_logs"][-1] if engine_output["turn_logs"] else {}
    return {
        "mode": "defense_demo",
        "sample_id": sample.sample_id,
        "title": sample.meta.get("title", sample.sample_id),
        "description": sample.meta.get("description", ""),
        "expected_action": sample.meta.get("expected_action", ""),
        "dataset_name": sample.meta.get("dataset_name", ""),
        "source_run": sample.meta.get("source_run", ""),
        "turns": [{"role": turn.role, "content": turn.content} for turn in sample.turns],
        "manual_logs": manual_logs,
        "engine_output": engine_output,
        "summary": {
            "prediction": engine_output["prediction"],
            "final_action": final_turn.get("action", "allow"),
            "final_risk_level": final_turn.get("risk_level", "low"),
            "final_fused_risk": final_turn.get("fused_risk", 0.0),
            "final_cumulative_risk": engine_output.get("final_cumulative_risk", 0.0),
        },
    }


def _localize_turn_log(log: Dict[str, Any]) -> Dict[str, Any]:
    localized = json.loads(json.dumps(log, ensure_ascii=False))
    localized["original_assistant"] = _translate_text(localized.get("original_assistant", ""))
    if "defended_assistant" in localized:
        localized["defended_assistant"] = _translate_text(localized.get("defended_assistant", ""))
    if "module_3_policy_decision" in localized:
        decision = localized["module_3_policy_decision"]
        decision["risk_level"] = _display_risk(decision.get("risk_level", ""))
        decision["action"] = _display_action(decision.get("action", ""))
        decision["strategy"] = _display_strategy(decision.get("strategy", ""))
        decision["defended_message"] = _translate_text(decision.get("defended_message", ""))
        decision["safety_notice"] = _translate_text(decision.get("safety_notice", ""))
        decision["preserved_excerpt"] = _translate_text(decision.get("preserved_excerpt", ""))
    if "action" in localized:
        localized["action"] = _display_action(localized.get("action", ""))
    if "risk_level" in localized:
        localized["risk_level"] = _display_risk(localized.get("risk_level", ""))
    if "strategy" in localized:
        localized["strategy"] = _display_strategy(localized.get("strategy", ""))
    if "safety_notice" in localized:
        localized["safety_notice"] = _translate_text(localized.get("safety_notice", ""))
    return localized


def _localize_engine_output(engine_output: Dict[str, Any]) -> Dict[str, Any]:
    localized = json.loads(json.dumps(engine_output, ensure_ascii=False))
    localized["turn_logs"] = [_localize_turn_log(log) for log in engine_output.get("turn_logs", [])]
    defended_turns = []
    for turn in localized.get("defended_turns", []):
        if turn.get("role") == "assistant":
            turn["content"] = _translate_text(turn.get("content", ""))
        defended_turns.append(turn)
    localized["defended_turns"] = defended_turns
    return localized


def localize_demo_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": result.get("mode", "defense_demo"),
        "sample_id": result.get("sample_id", ""),
        "title": _translate_text(result.get("title", "")),
        "description": _translate_text(result.get("description", "")),
        "expected_action": _display_action(result.get("expected_action", "")),
        "dataset_name": result.get("dataset_name", ""),
        "source_run": result.get("source_run", ""),
        "turns": result.get("turns", []),
        "manual_logs": [_localize_turn_log(log) for log in result.get("manual_logs", [])],
        "engine_output": _localize_engine_output(result.get("engine_output", {})),
        "summary": {
            "prediction": result.get("summary", {}).get("prediction", 0),
            "final_action": _display_action(result.get("summary", {}).get("final_action", "allow")),
            "final_risk_level": _display_risk(result.get("summary", {}).get("final_risk_level", "low")),
            "final_fused_risk": result.get("summary", {}).get("final_fused_risk", 0.0),
            "final_cumulative_risk": result.get("summary", {}).get("final_cumulative_risk", 0.0),
        },
    }


def localize_actor_result(result: Dict[str, Any]) -> Dict[str, Any]:
    localized = json.loads(json.dumps(result, ensure_ascii=False))
    localized["safety_notice"] = _translate_text(localized.get("safety_notice", ""))
    localized["defense_summary"]["final_action"] = _display_action(localized["defense_summary"].get("final_action", "allow"))
    rounds = []
    for round_item in localized.get("rounds", []):
        round_item["target_response"]["content"] = _translate_text(round_item["target_response"].get("content", ""))
        round_item["defense_response"]["content"] = _translate_text(round_item["defense_response"].get("content", ""))
        round_item["defense_log"] = _localize_turn_log(round_item.get("defense_log", {}))
        rounds.append(round_item)
    localized["rounds"] = rounds
    localized["defense_output"] = _localize_engine_output(localized.get("defense_output", {}))
    return localized


def build_custom_sample(user_text: str, assistant_text: str) -> DialogueSample:
    turns = [Turn(role="user", content=user_text.strip())]
    if assistant_text.strip():
        turns.append(Turn(role="assistant", content=assistant_text.strip()))
    return DialogueSample(
        sample_id="custom_demo_case",
        source="web_demo",
        label=None,
        turns=turns,
        meta={"title": "Custom sample", "description": "A manually entered web demo sample."},
    )


def save_result(result: Dict[str, Any], output_dir: Path, prefix: str = "web_demo") -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = output_dir / f"{prefix}_result_{ts}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"json_path": str(result_path)}


def build_app_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ADM-MRJA Web Demo</title>
  <style>
    :root { --bg:#f5f1e8; --panel:#fffdf9; --ink:#1f2937; --line:#e5e7eb; --accent:#0f766e; --accent2:#b45309; }
    body { margin:0; font-family:"Segoe UI","Microsoft YaHei",sans-serif; background:linear-gradient(180deg,#faf7f0,#f3eee3); color:var(--ink); }
    .shell { max-width:1440px; margin:0 auto; padding:24px; }
    .hero,.panel { background:var(--panel); border:1px solid #e6dece; border-radius:20px; box-shadow:0 10px 30px rgba(0,0,0,.06); }
    .hero { padding:24px; }
    .layout { display:grid; grid-template-columns:400px 1fr; gap:20px; margin-top:20px; }
    .panel { padding:18px; }
    .tabs { display:flex; gap:10px; margin-top:14px; }
    .tab-btn,button { border:0; border-radius:999px; padding:10px 16px; cursor:pointer; font-weight:700; }
    .tab-btn { background:#e8ecef; color:#334155; }
    .tab-btn.active { background:var(--accent); color:#fff; }
    .section { display:none; }
    .section.active { display:block; }
    .card { border:1px solid var(--line); border-radius:14px; padding:12px; margin-bottom:10px; background:#fbfcfd; cursor:pointer; }
    .card.active { border-color:var(--accent); background:#f0fdfa; }
    textarea { width:100%; min-height:90px; padding:12px; border-radius:12px; border:1px solid #d1d5db; resize:vertical; }
    .meta { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:14px 0; }
    .meta-card { background:#f8fafc; border:1px solid var(--line); border-radius:12px; padding:12px; }
    .timeline { display:grid; gap:14px; }
    .step { border-left:4px solid var(--accent); background:#fff; border-radius:14px; padding:14px; border:1px solid var(--line); }
    .bubble { border-radius:12px; padding:10px 12px; margin:8px 0; white-space:pre-wrap; }
    .user { background:#eef2ff; }
    .target { background:#fff7ed; }
    .defense { background:#ecfdf5; }
    pre { white-space:pre-wrap; word-break:break-word; background:#0f172a; color:#e2e8f0; border-radius:12px; padding:14px; overflow:auto; }
    .notice { padding:10px 12px; border-radius:12px; background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; margin:10px 0; }
    .hint { padding:10px 12px; border-radius:12px; background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; margin:10px 0; }
    .badge { display:inline-block; margin-right:8px; padding:4px 10px; border-radius:999px; background:#e2e8f0; color:#334155; font-size:12px; font-weight:700; }
    .subtle { color:#64748b; font-size:13px; line-height:1.5; }
    .mono { font-family:Consolas, Menlo, monospace; }
    @media (max-width: 960px) { .layout { grid-template-columns:1fr; } .meta { grid-template-columns:1fr 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <h1>ADM-MRJA Defense Demo</h1>
      <p>The preset cards below are automatically built from the latest single-turn benchmark run. They cover all five defense outcomes: allow, guide, rewrite, partial refusal, and full refusal.</p>
      <div class="tabs">
        <button class="tab-btn active" data-tab="demo">Single-turn presets</button>
        <button class="tab-btn" data-tab="actor">ActorAttack replay</button>
      </div>
    </div>
    <div class="layout">
      <div class="panel">
        <div id="demoSection" class="section active">
          <h2>Latest preset cases</h2>
          <div class="subtle">Each preset is a real sample pulled from the latest <span class="mono">singleturn_benchmarks_all_*.json</span> file when available.</div>
          <div id="demoCases" style="margin-top:12px;"></div>
          <h3>Custom sample</h3>
          <label>User input</label>
          <textarea id="userText" placeholder="Enter a test prompt"></textarea>
          <label>Original target-model reply</label>
          <textarea id="assistantText" placeholder="Enter the reply to be filtered"></textarea>
          <div style="display:flex; gap:10px; margin-top:12px;">
            <button style="background:var(--accent); color:#fff;" id="runCustomBtn">Run custom sample</button>
            <button style="background:var(--accent2); color:#fff;" id="refreshDemoBtn">Reload preset cases</button>
          </div>
        </div>
        <div id="actorSection" class="section">
          <h2>ActorAttack cases</h2>
          <div class="notice">This area is replay-only. The next attack turn comes from recorded ActorAttack dialogue, not from online generation.</div>
          <div id="actorCases"></div>
          <div style="display:flex; gap:10px; margin-top:12px;">
            <button style="background:var(--accent2); color:#fff;" id="refreshActorBtn">Reload ActorAttack cases</button>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2>Summary and trace</h2>
        <div id="summary"></div>
        <div id="content"></div>
      </div>
    </div>
  </div>
  <script>
    let currentTab = 'demo';
    let demoCases = [];
    let actorCases = [];
    let selectedDemoId = null;
    let selectedActorId = null;

    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.onclick = () => {
        currentTab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(x => x.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('demoSection').classList.toggle('active', currentTab === 'demo');
        document.getElementById('actorSection').classList.toggle('active', currentTab === 'actor');
      };
    });

    async function loadDemoCases() {
      const res = await fetch('/api/demo-cases');
      demoCases = await res.json();
      const root = document.getElementById('demoCases');
      root.innerHTML = '';
      demoCases.forEach(item => {
        const div = document.createElement('div');
        div.className = 'card' + (item.sample_id === selectedDemoId ? ' active' : '');
        div.innerHTML = `
          <strong>${item.title}</strong>
          <div style="margin-top:8px;">
            <span class="badge">Expected: ${item.expected_action_label || 'n/a'}</span>
            <span class="badge">${item.dataset_name || 'Unknown dataset'}</span>
          </div>
          <p>${item.description}</p>
          <div class="subtle">Source run: <span class="mono">${item.source_run || 'fallback_demo_cases'}</span></div>
        `;
        div.onclick = async () => {
          selectedDemoId = item.sample_id;
          await loadDemoCases();
          const res = await fetch('/api/demo-run', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode:'preset', sample_id:item.sample_id}) });
          renderDemo(await res.json());
        };
        root.appendChild(div);
      });
    }

    async function loadActorCases() {
      const res = await fetch('/api/actorattack-cases');
      actorCases = await res.json();
      const root = document.getElementById('actorCases');
      root.innerHTML = '';
      actorCases.forEach(item => {
        const div = document.createElement('div');
        div.className = 'card' + (item.case_id === selectedActorId ? ' active' : '');
        div.innerHTML = `<strong>${item.instruction || 'Untitled case'}</strong><p>${item.case_id}<br>Rounds: ${item.turn_count} / Attack model: ${item.attack_model} / Target model: ${item.target_model}</p>`;
        div.onclick = async () => {
          selectedActorId = item.case_id;
          await loadActorCases();
          const res = await fetch('/api/actorattack-run', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({case_id:item.case_id}) });
          renderActor(await res.json());
        };
        root.appendChild(div);
      });
    }

    function renderDemo(payload) {
      const result = payload.result;
      const s = result.summary;
      document.getElementById('summary').innerHTML = `
        <div class="meta">
          <div class="meta-card"><div>Final action</div><strong>${s.final_action}</strong></div>
          <div class="meta-card"><div>Risk level</div><strong>${s.final_risk_level}</strong></div>
          <div class="meta-card"><div>Fused risk</div><strong>${s.final_fused_risk}</strong></div>
          <div class="meta-card"><div>Cumulative risk</div><strong>${s.final_cumulative_risk}</strong></div>
        </div>
        <p><strong>Case title:</strong> ${result.title}</p>
        <p><strong>Description:</strong> ${result.description}</p>
        <p><strong>Expected preset action:</strong> ${result.expected_action || 'custom sample'}</p>
        <p><strong>Dataset:</strong> ${result.dataset_name || 'custom sample'}</p>
        <p><strong>Source benchmark run:</strong> <span class="mono">${result.source_run || 'manual input'}</span></p>
        <p><strong>Saved JSON:</strong> <span class="mono">${payload.output.json_path}</span></p>
      `;
      document.getElementById('content').innerHTML = '<h3>Module trace</h3>' + result.manual_logs.map(log => `
        <div class="step">
          <div><span class="badge">Action ${log.module_3_policy_decision.action}</span><span class="badge">Strategy ${log.module_3_policy_decision.strategy || 'n/a'}</span></div>
          <div class="bubble user"><strong>User</strong><br>${log.user_text}</div>
          <div class="bubble target"><strong>Original reply</strong><br>${log.original_assistant || 'None'}</div>
          <div class="bubble defense"><strong>Defended reply</strong><br>${log.module_3_policy_decision.defended_message}</div>
          ${log.module_3_policy_decision.safety_notice ? `<div class="notice"><strong>Safety note:</strong> ${log.module_3_policy_decision.safety_notice}</div>` : ''}
          ${log.module_3_policy_decision.preserved_excerpt ? `<div class="hint"><strong>Preserved public-safe content:</strong> ${log.module_3_policy_decision.preserved_excerpt}</div>` : ''}
          <div class="hint"><strong>Removed high-risk fragment count:</strong> ${log.module_3_policy_decision.removed_fragments_count ?? 0}</div>
          <pre>${JSON.stringify(log, null, 2)}</pre>
        </div>
      `).join('');
    }

    function renderActor(payload) {
      const result = payload.result;
      const s = result.defense_summary;
      document.getElementById('summary').innerHTML = `
        <div class="notice">${result.safety_notice}</div>
        <div class="meta">
          <div class="meta-card"><div>Attack model</div><strong>${result.attack_meta.attack_model_name}</strong></div>
          <div class="meta-card"><div>Target model</div><strong>${result.attack_meta.target_model_name}</strong></div>
          <div class="meta-card"><div>Final action</div><strong>${s.final_action}</strong></div>
          <div class="meta-card"><div>Cumulative risk</div><strong>${s.final_cumulative_risk}</strong></div>
        </div>
        <p><strong>Original instruction:</strong> ${result.instruction}</p>
        <p><strong>Saved JSON:</strong> <span class="mono">${payload.output.json_path}</span></p>
      `;
      document.getElementById('content').innerHTML = '<h3>Attack and defense timeline</h3><div class="timeline">' + result.rounds.map(round => `
        <div class="step">
          <strong>Round ${round.round_id}</strong>
          <div><span class="badge">Action ${round.defense_log.action}</span><span class="badge">Strategy ${round.defense_log.strategy || 'n/a'}</span></div>
          <div class="bubble user"><strong>ActorAttack prompt</strong><br>${round.attack_turn.content}</div>
          <div class="bubble target"><strong>Original target reply</strong><br>${round.target_response.content || 'None'}</div>
          <div class="bubble defense"><strong>Defense-layer reply</strong><br>${round.defense_response.content || 'None'}</div>
          ${round.defense_log.safety_notice ? `<div class="notice"><strong>Safety note:</strong> ${round.defense_log.safety_notice}</div>` : ''}
          ${round.defense_log.preserved_excerpt ? `<div class="hint"><strong>Preserved public-safe content:</strong> ${round.defense_log.preserved_excerpt}</div>` : ''}
          <div class="hint"><strong>Removed high-risk fragment count:</strong> ${round.defense_log.removed_fragments_count ?? 0}</div>
          <pre>${JSON.stringify(round.defense_log, null, 2)}</pre>
        </div>
      `).join('') + '</div>';
    }

    document.getElementById('runCustomBtn').onclick = async () => {
      const res = await fetch('/api/demo-run', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ mode:'custom', user_text:document.getElementById('userText').value, assistant_text:document.getElementById('assistantText').value })
      });
      renderDemo(await res.json());
    };
    document.getElementById('refreshDemoBtn').onclick = loadDemoCases;
    document.getElementById('refreshActorBtn').onclick = loadActorCases;
    loadDemoCases();
    loadActorCases();
  </script>
</body>
</html>"""


def make_handler(cases_path: Path, output_dir: Path, actorattack_dir: Path):
    loaded_cases = load_cases(cases_path)
    case_map = {item["sample_id"]: item for item in loaded_cases}
    actor_replay = ActorAttackReplay(actorattack_dir)

    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                self._write_html(build_app_html())
                return
            if route == "/api/demo-cases":
                self._write_json(
                    [
                        {
                            "sample_id": item["sample_id"],
                            "title": item["title"],
                            "description": item["description"],
                            "expected_action": item.get("expected_action", ""),
                            "expected_action_label": item.get("expected_action_label", ""),
                            "dataset_name": item.get("dataset_name", ""),
                            "source_run": item.get("source_run", ""),
                        }
                        for item in loaded_cases
                    ]
                )
                return
            if route == "/api/actorattack-cases":
                self._write_json(actor_replay.list_cases()[:20])
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            route = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if route == "/api/demo-run":
                if payload.get("mode") == "preset":
                    selected_item = case_map.get(payload.get("sample_id"))
                    if selected_item is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "Unknown preset sample")
                        return
                    selected = selected_item["sample"]
                else:
                    selected = build_custom_sample(payload.get("user_text", ""), payload.get("assistant_text", ""))
                result = localize_demo_result(analyze_sample(selected))
                output = save_result(result, output_dir, prefix="web_demo")
                self._write_json({"result": result, "output": output})
                return
            if route == "/api/actorattack-run":
                raw_result = actor_replay.replay_case(payload["case_id"])
                result = localize_actor_result(raw_result)
                output = save_result(result, output_dir, prefix="actorattack_web_demo")
                self._write_json({"result": result, "output": output})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format, *args):
            return

    return Handler


def main():
    args = build_parser().parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(Path(args.cases_path), Path(args.output_dir), Path(args.actorattack_dir)),
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Web demo started at: {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
