"""A node = Core (deterministic: duties, tool execution, Policy Guard) + Mastermind
(LLM episode loop). Design doc 04, ch. 1 and 3.

The Mastermind never touches the board directly: every write goes through
`_exec_tool` → Policy Guard → Board, and every external text reaches the model only
inside a <data> block.
"""

from __future__ import annotations

import heapq
import itertools
import json
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from aichain.board import BoardError, BountyState, commitment_of
from aichain.common import data_block, escape_data, fmt_P, to_mP
from aichain.gateways.llm.base import (
    BudgetExhausted,
    CallLimitReached,
    LLMRequest,
    Message,
    ProviderError,
    TokenBucket,
)
from aichain.mastermind import prompt
from aichain.mastermind.policy import PolicyGuard, PolicyViolation
from aichain.mastermind.tools import (
    TOOLS,
    WRITE_TOOLS,
    AnswerArgs,
    AskArgs,
    GetContentArgs,
    GetStateArgs,
    JudgeArgs,
    RecallArgs,
    RememberArgs,
    tool_specs,
)
from aichain.memory import Memory

if TYPE_CHECKING:
    from aichain.sim.network import Network

# Placeholder until reputation is computed from rep_events (03 doc 4.3); new nodes start ≈ 0.45.
REP_PLACEHOLDER = 0.45

PRIORITY = {
    "judge_due": 1,
    "dispute_against_me": 1,
    "proposal_open": 2,
    "bounty_new": 3,
    "answer_revealed": 3,
    "bridge_question": 3,
    "peer_msg": 4,
    "idle_tick": 5,
}


@dataclass
class NodeConfig:
    node_id: str
    persona: str  # seeker | artisan | warden | nomad
    provider: str = "scripted"
    model: str = "scripted"
    temperature: float | None = 0.4
    max_output_tokens: int | None = 8000
    daily_budget_tokens: int = 2_000_000
    max_steps: int = 8
    max_episode_tokens: int = 40_000
    tags_interest: list[str] = field(default_factory=list)  # "*" = every tag


@dataclass(order=True)
class QueuedEvent:
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)


@dataclass
class EpisodeResult:
    node_id: str
    event: str
    steps: int
    llm_calls: int
    tokens: int
    summary: str | None
    stopped: str  # done | text | max_steps | max_tokens | budget | call_limit | provider_error


class Node:
    def __init__(self, cfg: NodeConfig, net: Network) -> None:
        self.cfg = cfg
        self.id = cfg.node_id
        self.net = net
        self.persona_layer = prompt.load(cfg.persona)
        self.core_prompt = prompt.load("core")
        self.memory = Memory()
        self.policy = PolicyGuard(cfg.node_id)
        self.bucket = TokenBucket(cfg.daily_budget_tokens, clock=net.clock_minutes)
        net.gateway.register(cfg.node_id, self.bucket)
        self._queue: list[QueuedEvent] = []
        self._seq = itertools.count()
        # answer_id -> (body, salt): kept locally until the Core reveals it.
        self._sealed: dict[str, tuple[dict[str, Any], str]] = {}

    # ---- events ----

    def enqueue(self, kind: str, payload: dict[str, Any]) -> None:
        heapq.heappush(self._queue, QueuedEvent(PRIORITY.get(kind, 4), next(self._seq), kind, payload))

    @property
    def has_pending(self) -> bool:
        return bool(self._queue)

    def run_pending(self) -> list[EpisodeResult]:
        results = []
        while self._queue:
            ev = heapq.heappop(self._queue)
            if not self.bucket.allows(ev.priority):
                self.net.log(self.id, "skip_event", kind=ev.kind,
                             reason=f"budget {self.bucket.state}")
                continue
            results.append(self.run_episode(ev))
        return results

    # ---- Core duties (no LLM) ----

    def run_duties(self) -> None:
        """Reveal committed answers as soon as their bounty enters REVEAL (04 doc 6.1)."""
        for answer_id, (body, salt) in list(self._sealed.items()):
            ans = self.net.board.answers[answer_id]
            if self.net.board.bounty(ans.bounty_id).state != BountyState.REVEAL:
                continue
            body_hash = self.net.put_content(body, author=self.id)
            self.net.board.reveal_answer(self.id, answer_id, body_hash, salt)
            del self._sealed[answer_id]
            self.net.log(self.id, "duty_reveal", answer_id=answer_id)

    def on_settled(self, payload: dict[str, Any]) -> None:
        b = self.net.board.bounty(payload["bounty_id"])
        acct = self.net.board.account(self.id)
        accepted = self.net.board.answers[b.accepted].author if b.accepted else None
        text = (f"현상금 {b.bounty_id} 종료: {payload['outcome']}. 질문자 {b.asker}, "
                f"현상금 {fmt_P(b.reward_mP)}"
                + (f", 수락된 답변자 {accepted}" if accepted else "")
                + f". 내 잔액 {fmt_P(acct.balance_mP)} (잠김 {fmt_P(acct.locked_mP)}).")
        self.memory.remember(self.net.board.height, "episode", f"chain:{b.bounty_id}", 1.0, text,
                             tags=b.tags)

    # ---- Mastermind ----

    def _system_messages(self, event_text: str) -> list[Message]:
        board = self.net.board
        acct = board.account(self.id)
        core = prompt.render_core(
            self.core_prompt,
            node_id=self.id,
            hourly_limit=fmt_P(self.policy.hourly_limit_mP(acct.available_mP)).rstrip("P"),
            bounty_cap=fmt_P(self.policy.bounty_cap_mP(acct.available_mP)).rstrip("P"),
            budget_remaining=str(self.bucket.remaining()),
            budget_state=str(self.bucket.state),
            balance=fmt_P(acct.balance_mP).rstrip("P"),
            reputation=f"{REP_PLACEHOLDER:.2f}",
            persona_layer=self.persona_layer,
        )
        open_bounties = [b for b in board.bounties.values() if b.state == BountyState.OPEN]
        to_judge = [b for b in board.bounties.values()
                    if b.asker == self.id and b.state == BountyState.JUDGING]
        state = (
            "## 현재 상태\n"
            f"- 블록 높이 {board.height}\n"
            f"- 잔액 {fmt_P(acct.balance_mP)}, 잠김 {fmt_P(acct.locked_mP)}, "
            f"사용 가능 {fmt_P(acct.available_mP)}\n"
            f"- 평판 {REP_PLACEHOLDER:.2f} (프로토타입: 고정값)\n"
            f"- LLM 예산 {self.bucket.remaining()} 토큰, 상태 {self.bucket.state}\n"
            f"- 열린 현상금 {len(open_bounties)}개, 내가 판정할 현상금 {len(to_judge)}개\n\n"
            "## 이 프로토타입에서 쓸 수 있는 도구\n"
            f"{', '.join(TOOLS)}. 고정 코어에 적힌 나머지 도구는 아직 없다.\n"
            "할 일을 마치면 done을 호출해 에피소드를 끝낸다."
        )
        memories = self.memory.recall(event_text, k=8)
        if memories:
            mem_text = "## 관련 기억\n" + "\n".join(
                f"- #{m.seq} [{m.kind}] source={m.source} trust={m.trust:.2f}: "
                f"{escape_data(m.content)}" for m in memories)
        else:
            mem_text = "## 관련 기억\n(없음)"
        return [Message("system", core), Message("system", state), Message("system", mem_text)]

    def render_event(self, ev: QueuedEvent) -> str:
        p = ev.payload
        board = self.net.board
        if ev.kind == "bridge_question":
            q = {k: p[k] for k in ("title", "body", "acceptance_criteria")}
            inner = data_block(f"bridge:{p['client']}", json.dumps(q, ensure_ascii=False, indent=1))
            attrs = f'client="{escape_data(p["client"])}"'
        elif ev.kind == "bounty_new":
            b = board.bounty(p["bounty_id"])
            q = self.net.content.get(b.question_hash)
            inner = data_block(f"peer:{b.asker}", json.dumps(q, ensure_ascii=False, indent=1),
                               trust=REP_PLACEHOLDER)
            attrs = (f'bounty_id="{b.bounty_id}" asker="{b.asker}" reward="{fmt_P(b.reward_mP)}" '
                     f'tags="{",".join(b.tags)}" commit_until="{b.commit_until}"')
        elif ev.kind == "judge_due":
            b = board.bounty(p["bounty_id"])
            q = self.net.content.get(b.question_hash)
            parts = [data_block(f"self:{b.bounty_id}", json.dumps(q, ensure_ascii=False, indent=1))]
            for aid in b.answers:
                a = board.answers[aid]
                if not a.revealed:
                    continue
                body = self.net.content.get(a.body_hash)
                parts.append(f'<answer id="{a.answer_id}" author="{a.author}">\n'
                             + data_block(f"peer:{a.author}",
                                          json.dumps(body, ensure_ascii=False, indent=1),
                                          trust=REP_PLACEHOLDER)
                             + "\n</answer>")
            inner = "\n".join(parts)
            attrs = (f'bounty_id="{b.bounty_id}" reward="{fmt_P(b.reward_mP)}" '
                     f'judge_until="{b.judge_until}"')
        else:
            inner = data_block("core", json.dumps(p, ensure_ascii=False))
            attrs = ""
        return f'<event type="{ev.kind}" {attrs}>\n{inner}\n</event>'

    def run_episode(self, ev: QueuedEvent) -> EpisodeResult:
        cfg = self.cfg
        self.policy.start_episode()
        event_text = self.render_event(ev)
        messages = self._system_messages(event_text) + [Message("user", event_text)]
        tools = tool_specs()
        steps = calls = tokens = 0
        summary: str | None = None
        stopped = "max_steps"
        self.net.log(self.id, "episode_start", kind=ev.kind)
        while steps < cfg.max_steps:
            request = LLMRequest(self.id, cfg.model, messages, tools, cfg.temperature,
                                 cfg.max_output_tokens)
            try:
                resp = self.net.gateway.chat(cfg.provider, request, ev.priority)
            except BudgetExhausted as e:
                stopped, summary = "budget", str(e)
                break
            except CallLimitReached as e:
                stopped, summary = "call_limit", str(e)
                break
            except ProviderError as e:
                stopped, summary = "provider_error", str(e)
                break
            calls += 1
            tokens += resp.usage.input_tokens + resp.usage.output_tokens
            messages.append(Message("assistant", resp.text, resp.tool_calls, raw=resp.raw))
            if resp.text:
                self.net.log(self.id, "llm_text", text=resp.text[:500])
            if not resp.tool_calls:
                stopped, summary = "text", resp.text
                break
            finished = False
            for call in resp.tool_calls:
                steps += 1
                if call.name == "done":
                    summary = _parse_summary(call.arguments)
                    result = "에피소드 종료."
                    finished = True
                else:
                    result = self._exec_tool(call.name, call.arguments)
                messages.append(Message("tool", result, call_id=call.call_id, name=call.name))
                if finished or steps >= cfg.max_steps:
                    break
            if finished:
                stopped = "done"
                break
            if tokens >= cfg.max_episode_tokens:
                stopped = "max_tokens"
                break
        self.memory.remember(self.net.board.height, "episode", "self", 1.0,
                             f"[{ev.kind}] {summary or '(요약 없음)'}")
        result = EpisodeResult(self.id, ev.kind, steps, calls, tokens, summary, stopped)
        self.net.log(self.id, "episode_end", kind=ev.kind, steps=steps, llm_calls=calls,
                     tokens=tokens, stopped=stopped, summary=summary)
        return result

    # ---- tool execution (Core) ----

    def _exec_tool(self, name: str, raw_args: str) -> str:
        if name not in TOOLS:
            return data_block("core", f"오류: 알 수 없는 도구 {name}")
        model_cls = TOOLS[name][0]
        try:
            args = model_cls.model_validate_json(raw_args or "{}")
        except ValidationError as e:
            self.net.log(self.id, "tool_invalid", tool=name, error=str(e)[:300])
            return data_block("core", f"오류: 인자 검증 실패\n{e}")
        try:
            if name in WRITE_TOOLS:
                self.policy.check_write()
            out = self._dispatch(name, args)
        except PolicyViolation as e:
            self.net.log(self.id, "policy_reject", tool=name, reason=str(e))
            self.memory.remember(self.net.board.height, "episode", "self", 1.0,
                                 f"Policy Guard가 {name} 호출을 거부함: {e}")
            return data_block("core", f"거부(Policy Guard): {e}")
        except BoardError as e:
            self.net.log(self.id, "board_reject", tool=name, reason=str(e))
            return data_block("core", f"거부(프로토콜): {e}")
        self.net.log(self.id, "tool", tool=name, args=json.loads(args.model_dump_json()),
                     result=out[:300])
        return data_block("core", out)

    def _dispatch(self, name: str, args: BaseModel) -> str:
        board = self.net.board
        acct = board.account(self.id)
        h = board.height
        if isinstance(args, GetStateArgs):
            return self._state_text(args.what)
        if isinstance(args, GetContentArgs):
            obj = self.net.content.get(args.hash)
            if obj is None:
                return f"콘텐츠 {args.hash} 없음"
            author = self.net.content_author.get(args.hash, "unknown")
            return f"source=peer:{author}\n" + json.dumps(obj, ensure_ascii=False, indent=1)
        if isinstance(args, RecallArgs):
            recs = self.memory.recall(args.query, args.k)
            if not recs:
                return "관련 기억 없음"
            return "\n".join(f"#{r.seq} [{r.kind}] source={r.source} trust={r.trust:.2f}: {r.content}"
                             for r in recs)
        if isinstance(args, RememberArgs):
            rec = self.memory.remember(h, args.kind, "self", 1.0, args.content, args.tags, args.refs)
            return f"기억 #{rec.seq} 기록됨"
        if isinstance(args, AskArgs):
            reward = to_mP(args.reward_P)
            self.policy.check_spend(reward, acct.available_mP, h, bounty=True)
            q_hash = self.net.put_content(args.question.model_dump(), author=self.id)
            b = board.create_bounty(self.id, q_hash, reward, args.tags, args.max_answers)
            self.policy.record_write(reward, h)
            self.net.announce_bounty(b.bounty_id)
            return (f"현상금 생성: bounty_id={b.bounty_id}, 현상금 {fmt_P(reward)} 잠김, "
                    f"커밋 마감 블록 {b.commit_until}, 판정 마감 블록 {b.judge_until}")
        if isinstance(args, AnswerArgs):
            body = args.answer.model_dump()
            salt = secrets.token_hex(16)
            body_hash = self.net.content.hash_of(body)
            ans = board.commit_answer(self.id, args.bounty_id, commitment_of(body_hash, salt, self.id))
            self._sealed[ans.answer_id] = (body, salt)
            self.policy.record_write(0, h)
            return (f"답변 커밋됨: answer_id={ans.answer_id}, 보증금 "
                    f"{fmt_P(board.params.commit_bond_mP)} 잠김. 공개는 Core가 자동으로 한다.")
        if isinstance(args, JudgeArgs):
            if not args.accept and not args.reject:
                raise BoardError("accept 또는 reject 중 하나는 있어야 한다")
            done: list[str] = []
            for r in args.reject:
                board.reject_answer(self.id, args.bounty_id, r.answer_id, r.reason)
                done.append(f"거절 {r.answer_id}")
            if args.accept:
                events = board.accept_answer(self.id, args.bounty_id, args.accept)
                self.net.dispatch(events)
                done.append(f"수락 {args.accept} → 현상금 지급(분쟁 기간 동안 잠김)")
            self.policy.record_write(0, h)
            return "; ".join(done)
        raise BoardError(f"unhandled tool {name}")

    def _state_text(self, what: str) -> str:
        board = self.net.board
        acct = board.account(self.id)
        if what == "balance":
            return (f"잔액 {fmt_P(acct.balance_mP)}, 잠김 {fmt_P(acct.locked_mP)}, "
                    f"사용 가능 {fmt_P(acct.available_mP)}, "
                    f"최근 1시간 지출 {fmt_P(self.policy.spent_last_hour(board.height))}, "
                    f"LLM 예산 {self.bucket.remaining()} ({self.bucket.state})")
        rows = []
        for b in board.bounties.values():
            mine = b.asker == self.id or any(board.answers[a].author == self.id for a in b.answers)
            if what == "bounties" and b.state != BountyState.OPEN:
                continue
            if what == "my_bounties" and not mine:
                continue
            q = self.net.content.get(b.question_hash) or {}
            rows.append(f"- {b.bounty_id} [{b.state}] 질문자 {b.asker}, {fmt_P(b.reward_mP)}, "
                        f"tags={','.join(b.tags)}, 답변 {len(b.answers)}/{b.max_answers}, "
                        f"제목: {escape_data(str(q.get('title', '')))}")
        return "\n".join(rows) or "(없음)"


def _parse_summary(raw: str) -> str | None:
    try:
        return json.loads(raw or "{}").get("summary")
    except (json.JSONDecodeError, AttributeError):
        return raw[:500] if raw else None
