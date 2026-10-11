"""Initial setup: choose each node's research interests.

    python -m aichain.setup                         # interactive, defaults shown
    python -m aichain.setup --defaults              # write the defaults
    python -m aichain.setup --node n2 --interest "Rust: 시스템 프로그래밍 언어 동향" --interest "WebAssembly"
    python -m aichain.setup --show

Saved to data/nodes.json; the demo and the bridge read it. Interests go into the node's
persona layer (not the fixed core), and research episodes pick topics from them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_FILE = Path("data/nodes.json")

DEFAULT_INTERESTS: dict[str, list[dict[str, str]]] = {
    "n1": [
        {"topic": "AI 에이전트 동향", "description": "LLM 에이전트 프레임워크, 도구 사용, 멀티 에이전트 협업의 최신 흐름"},
        {"topic": "LLM 모델 출시", "description": "새로 나온 모델과 벤치마크, 가격 변화"},
    ],
    "n2": [
        {"topic": "Python 개발", "description": "Python 언어·표준 라이브러리 변화, 성능, 패키징"},
        {"topic": "소프트웨어 엔지니어링 실무", "description": "테스트, 디버깅, 코드 품질 도구"},
    ],
    "n3": [
        {"topic": "보안 위협", "description": "새 취약점, 공급망 공격, 프롬프트 인젝션"},
        {"topic": "사기·어뷰징 탐지", "description": "온라인 사기 수법과 담합·어뷰징 탐지 기법"},
    ],
    "n4": [
        {"topic": "과학 소식", "description": "물리·생물·우주 등 주목할 만한 연구 결과"},
        {"topic": "세계 경제", "description": "주요 경제 지표와 산업 흐름"},
    ],
}


def parse_interest(text: str) -> dict[str, str]:
    topic, _, desc = text.partition(":")
    return {"topic": topic.strip(), "description": desc.strip()}


def load(path: Path = DEFAULT_FILE) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {k: list(v) for k, v in DEFAULT_INTERESTS.items()}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {nid: cfg.get("interests", []) for nid, cfg in raw.get("nodes", {}).items()}


def save(interests: dict[str, list[dict[str, str]]], path: Path = DEFAULT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"nodes": {nid: {"interests": items} for nid, items in sorted(interests.items())}}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def interactive(current: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    print("노드별 관심사를 정합니다. 형식: '주제: 설명'. 빈 줄이면 현재 값을 유지합니다.\n")
    out = {}
    for nid in sorted(current):
        print(f"[{nid}] 현재: " + "; ".join(i["topic"] for i in current[nid]))
        entered = []
        while True:
            line = input(f"  {nid} 관심사 {len(entered) + 1} (끝내려면 Enter): ").strip()
            if not line:
                break
            entered.append(parse_interest(line))
        out[nid] = entered or current[nid]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Set each node's research interests")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--node", help="node to set (with --interest)")
    ap.add_argument("--interest", action="append", default=[], help="'topic: description'")
    ap.add_argument("--defaults", action="store_true", help="write the default interests")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    current = load(args.file)
    if args.show:
        print(json.dumps(current, ensure_ascii=False, indent=2))
        return 0
    if args.defaults:
        current = {k: list(v) for k, v in DEFAULT_INTERESTS.items()}
    elif args.node:
        if not args.interest:
            ap.error("--node needs at least one --interest")
        current[args.node] = [parse_interest(i) for i in args.interest]
    else:
        current = interactive(current)
    save(current, args.file)
    print(f"saved {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
