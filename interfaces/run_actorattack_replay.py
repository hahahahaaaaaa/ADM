import argparse
import json
from pathlib import Path

from actorattack_replay import ActorAttackReplay

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "framework_outputs"
DEFAULT_ACTORATTACK_DIR = ROOT_DIR / "attack_methods" / "ActorAttack" / "attack_result"


def build_parser():
    parser = argparse.ArgumentParser(description="运行 ActorAttack 安全回放验证")
    parser.add_argument("--actorattack-dir", default=str(DEFAULT_ACTORATTACK_DIR), help="ActorAttack attack_result 目录")
    parser.add_argument("--case-id", default="", help="指定 case_id；为空则自动选择第一条")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR), help="输出目录")
    return parser


def main():
    args = build_parser().parse_args()
    replay = ActorAttackReplay(Path(args.actorattack_dir))
    cases = replay.list_cases()
    if not cases:
        raise RuntimeError("未发现可回放的 ActorAttack 结果文件")

    case_id = args.case_id or cases[0]["case_id"]
    result = replay.replay_case(case_id, Path(args.output_dir))
    print("ActorAttack 安全回放完成")
    print(f"Case: {case_id}")
    print(f"Saved: {result.get('saved_path', '')}")
    print(json.dumps(result["defense_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
