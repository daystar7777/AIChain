"""M7 analyses (08 doc M7): collusion between node pairs, persona convergence, point
concentration and vote correlation, plus answer competition, research vs acceptance, velocity
and issuance/burn by day (03 doc 2.4, 04 doc 5.4, 05 doc 5, 07 doc T4).

Inputs are read-only: a Board replayed from a copy of the chain log (`load_chain`), the
journal (metrics series and tool entries) and the knowledge store's persona history.
`render_markdown` turns the result into the report's analysis section.

Payments are read from the board's reputation events, which name the payee and (rules 2) the
payer; before rules 2 the payer of an answer reward is the bounty's asker. A payout clawed back
by an upheld dispute still counts as paid here.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from aichain.board import Board
from aichain.metrics import gini, vote_agreement

NODES = ("n1", "n2", "n3", "n4")
PERSONAS = {"n1": "seeker", "n2": "artisan", "n3": "warden", "n4": "nomad"}
# Reputation events that move value from a payer to the event's account. Loan principal and
# repayment are "flow" events; loan_interest and income_credit would count the same value twice.
FLOW_KINDS = ("answer_accepted", "answer_auto", "tool_licensed", "tool_called", "info_sold",
              "transfer_in", "flow")
# Tools that look something up before answering.
RESEARCH_TOOLS = ("search", "fetch", "offers", "offer_buy", "recall", "get_content", "tool_run")
TIME_STEP = 60  # blocks between the (height, time) points read from the chain log

# Flag thresholds for the pair table (descriptive, not a verdict).
MIN_VOLUME_mP = 10_000
RECIPROCITY_FLAG = 0.5
INCOME_SHARE_FLAG = 0.6
EXCESS_SHARE_FLAG = 0.3  # above the payer's share of all node spending (n1 asks most questions)
ACCEPT_BIAS_FLAG = 0.3
MIN_ANSWERS = 3


class Clock:
    """Maps block heights to wall-clock time from (height, ts) points of the chain log."""

    def __init__(self, points: list[tuple[int, float]]) -> None:
        self.heights = [h for h, _ in points]
        self.times = [t for _, t in points]

    def ts(self, height: int) -> float | None:
        if not self.heights:
            return None
        i = bisect.bisect_right(self.heights, height) - 1
        return self.times[max(i, 0)]

    def height_at(self, ts: float) -> int:
        """First recorded height at or after `ts` (0 when there are no points)."""
        i = bisect.bisect_left(self.times, ts)
        return self.heights[i] if i < len(self.heights) else (self.heights[-1] + 1 if self.heights else 0)

    def day(self, height: int) -> str:
        ts = self.ts(height)
        return time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "?"


def load_chain(db: Path, periods: str = "short") -> tuple[Board, Clock]:
    from aichain.chain.verify import replay_copy, start_params

    board, _, times = replay_copy(db, start_params(periods), times_every=TIME_STEP)
    return board, Clock(times)


# ---- payments between accounts ----

def _payer(board: Board, e: dict[str, Any]) -> str | None:
    if e.get("payer"):
        return e["payer"]
    if e["kind"] in ("answer_accepted", "answer_auto") and e.get("bounty_id") in board.bounties:
        return board.bounties[e["bounty_id"]].asker
    if e["kind"] == "transfer_in":
        return e.get("sender")
    return None


def flows(board: Board, since_h: int = 0) -> list[dict[str, Any]]:
    out = []
    for e in board.rep_events:
        if e["kind"] in FLOW_KINDS and e.get("value_mP") and e["height"] >= since_h:
            out.append({"height": e["height"], "payer": _payer(board, e), "payee": e["account"],
                        "kind": e["kind"], "mP": e["value_mP"]})
    return out


def _settled_bounties(board: Board, since_h: int) -> list[Any]:
    return [b for b in board.bounties.values() if b.settled_at is not None and b.created_at >= since_h]


def _authors(board: Board, b: Any) -> list[str]:
    return sorted({board.answers[a].author for a in b.answers if a in board.answers and board.answers[a].revealed})


def _accepted_author(board: Board, b: Any) -> str | None:
    a = board.answers.get(b.accepted) if b.accepted else None
    return a.author if a else None


def pairs(board: Board, fl: list[dict[str, Any]], since_h: int = 0,
          nodes: tuple[str, ...] = NODES) -> list[dict[str, Any]]:
    """One row per node pair: payments both ways, how much of each side's node income came from
    the other, the asker's acceptance of the other's answers against that answerer's rate
    elsewhere, ratings both ways and vote agreement. `flags` names the patterns 07 doc T4 warns
    about (wash trading, captive income, favouritism), with the thresholds above."""
    vol: Counter = Counter()
    trades: Counter = Counter()
    income: Counter = Counter()
    spent: Counter = Counter()
    for f in fl:
        if f["payer"] in nodes and f["payee"] in nodes:
            vol[(f["payer"], f["payee"])] += f["mP"]
            trades[(f["payer"], f["payee"])] += 1
            income[f["payee"]] += f["mP"]
            spent[f["payer"]] += f["mP"]
    total = sum(spent.values())
    answered: Counter = Counter()
    accepted: Counter = Counter()
    for b in _settled_bounties(board, since_h):
        winner = _accepted_author(board, b)
        for au in _authors(board, b):
            answered[(b.asker, au)] += 1
            accepted[(b.asker, au)] += au == winner
    r_sum: Counter = Counter()
    r_n: Counter = Counter()
    for r in board.ratings.values():
        if not r.void and r.height >= since_h:
            r_sum[(r.rater, r.subject)] += r.score
            r_n[(r.rater, r.subject)] += 1
    votes = vote_agreement([p for p in board.proposals.values() if p.created_height >= since_h])

    def accept_rate(asker: str, author: str) -> tuple[int, float | None, float | None]:
        n = answered[(asker, author)]
        here = accepted[(asker, author)] / n if n else None
        n_else = sum(v for (a, au), v in answered.items() if au == author and a != asker)
        k_else = sum(v for (a, au), v in accepted.items() if au == author and a != asker)
        return n, here, (k_else / n_else if n_else else None)

    def mean(key: tuple[str, str]) -> float | None:
        return round(r_sum[key] / r_n[key], 2) if r_n[key] else None

    rows = []
    for a, b in combinations(nodes, 2):
        ab, ba = vol[(a, b)], vol[(b, a)]
        hi = max(ab, ba)
        row: dict[str, Any] = {
            "pair": f"{a}-{b}", "a_to_b_mP": ab, "b_to_a_mP": ba, "trades": trades[(a, b)] + trades[(b, a)],
            "reciprocity": round(min(ab, ba) / hi, 2) if hi else 0.0,
            "b_income_share_from_a": round(ab / income[b], 2) if income[b] else 0.0,
            "a_income_share_from_b": round(ba / income[a], 2) if income[a] else 0.0,
            "rating_a_to_b": mean((a, b)), "rating_b_to_a": mean((b, a)),
            "votes_agree": votes.get(f"{a}-{b}"),
        }
        flags = []
        if ab + ba >= MIN_VOLUME_mP and row["reciprocity"] >= RECIPROCITY_FLAG:
            flags.append("주고받기")
        for payee, payer, share in ((b, a, row["b_income_share_from_a"]), (a, b, row["a_income_share_from_b"])):
            # The share the payer would have if every node spread its spending evenly: its part
            # of what all the other nodes spent (the payee cannot pay itself). A lone asker
            # paying everyone is not captive income; one node paying far above its weight is.
            others = total - spent[payee]
            expected = spent[payer] / others if others else 0.0
            if (vol[(payer, payee)] >= MIN_VOLUME_mP and share >= INCOME_SHARE_FLAG
                    and share - expected >= EXCESS_SHARE_FLAG):
                flags.append(f"{payee} 수입의 {share:.0%}가 {payer}에게서")
        for asker, author in ((a, b), (b, a)):
            n, here, elsewhere = accept_rate(asker, author)
            row[f"accept_{asker}_of_{author}"] = (n, None if here is None else round(here, 2),
                                                  None if elsewhere is None else round(elsewhere, 2))
            if n >= MIN_ANSWERS and here is not None and elsewhere is not None and here - elsewhere >= ACCEPT_BIAS_FLAG:
                flags.append(f"{asker}가 {author}의 답을 더 잘 채택")
        if (row["rating_a_to_b"] or 0) >= 1.5 and (row["rating_b_to_a"] or 0) >= 1.5:
            flags.append("서로 높은 평점")
        row["flags"] = flags
        rows.append(row)
    return rows


def competition(board: Board, since_h: int = 0, nodes: tuple[str, ...] = NODES) -> dict[str, Any]:
    """How many nodes answer each question, and who answers and wins."""
    per_bounty: Counter = Counter()
    answers: Counter = Counter()
    wins: Counter = Counter()
    asked: Counter = Counter()
    outcomes: Counter = Counter()
    settled = _settled_bounties(board, since_h)
    for b in settled:
        authors = _authors(board, b)
        per_bounty[min(len(authors), 3)] += 1
        asked[b.asker] += 1
        outcomes[str(b.outcome or b.state)] += 1
        answers.update(authors)
        w = _accepted_author(board, b)
        if w:
            wins[w] += 1
    return {"bounties": len(settled), "answerers_per_bounty": {("3+" if k == 3 else str(k)): v
                                                               for k, v in sorted(per_bounty.items())},
            "asked": {n: asked[n] for n in nodes}, "answers": {n: answers[n] for n in nodes},
            "wins": {n: wins[n] for n in nodes}, "outcomes": dict(outcomes)}


# ---- persona convergence ----

def _grams(text: str, n: int = 3) -> Counter:
    t = " ".join(text.split())
    return Counter(t[i:i + n] for i in range(max(len(t) - n + 1, 0)))


def similarity(a: str, b: str) -> float:
    """Cosine similarity of character trigram counts (works for Korean without a tokenizer)."""
    ga, gb = _grams(a), _grams(b)
    dot = sum(v * gb[k] for k, v in ga.items())
    na, nb = math.sqrt(sum(v * v for v in ga.values())), math.sqrt(sum(v * v for v in gb.values()))
    return round(dot / (na * nb), 4) if na and nb else 0.0


def persona_timeline(base: dict[str, str], rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[tuple[float, str]]]:
    """(ts, text) per node: the base persona at time 0, then every approved edit or rollback."""
    out: dict[str, list[tuple[float, str]]] = {}
    for n, text in base.items():
        line = [(0.0, text)]
        for r in rows.get(n, []):
            d = r["data"]
            if d.get("verdict") in ("approved", "rollback") and d.get("text"):
                line.append((r["ts"], d["text"]))
        out[n] = line
    return out


def _text_at(line: list[tuple[float, str]], ts: float) -> str:
    text = line[0][1]
    for t, x in line:
        if t <= ts:
            text = x
    return text


def convergence(timeline: dict[str, list[tuple[float, str]]], start: float, end: float,
                days: list[float] | None = None) -> dict[str, Any]:
    """Pairwise persona similarity at the period's start and end (and at each `days` point):
    rising similarity means the nodes are drifting towards one personality (04 doc 5.4)."""
    nodes = sorted(timeline)

    def at(ts: float) -> dict[str, float]:
        texts = {n: _text_at(timeline[n], ts) for n in nodes}
        return {f"{a}-{b}": similarity(texts[a], texts[b]) for a, b in combinations(nodes, 2)}

    s, e = at(start), at(end)
    mean = lambda d: round(sum(d.values()) / len(d), 4) if d else 0.0  # noqa: E731
    edits = {n: sum(1 for t, _ in timeline[n][1:] if start <= t <= end) for n in nodes}
    drift = {n: round(1 - similarity(_text_at(timeline[n], start), _text_at(timeline[n], end)), 4) for n in nodes}
    return {"start": s, "end": e, "mean_start": mean(s), "mean_end": mean(e), "edits": edits, "drift": drift,
            "series": [(d, mean(at(d))) for d in (days or [])]}


# ---- time series from the journal ----

def metrics_by_day(journal: sqlite3.Connection, since: float | None, nodes: tuple[str, ...] = NODES
                   ) -> list[dict[str, Any]]:
    """The last metrics row of each day: supply, issuance and burn so far, concentration. The
    day's issuance and burn are measured from the day before, or for the first day from the
    last row before `since` (else that day's first row)."""
    first: dict[str, dict[str, Any]] = {}
    last: dict[str, dict[str, Any]] = {}
    where, args = ("WHERE ts >= ?", (since,)) if since else ("", ())
    for ts, data in journal.execute(f"SELECT ts, data FROM metrics {where} ORDER BY ts", args):
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        last[day] = json.loads(data)
        first.setdefault(day, last[day])
    before = journal.execute("SELECT data FROM metrics WHERE ts < ? ORDER BY ts DESC LIMIT 1",
                             (since,)).fetchone() if since else None
    start = json.loads(before[0]) if before else (first[min(first)] if first else None)
    out, prev = [], (None if start is None else
                     {"issued_mP": start["supply"]["issued_mP"], "burned_mP": start["supply"]["burned_mP"]})
    for day in sorted(last):
        m = last[day]
        bal = {n: m["balances_mP"].get(n, 0) for n in nodes}
        total = sum(bal.values())
        row = {"day": day, "height": m["height"], "circulating_mP": m["supply"]["circulating_mP"],
               "issued_mP": m["supply"]["issued_mP"], "burned_mP": m["supply"]["burned_mP"],
               "gini": m.get("gini", gini(list(bal.values()))),
               "top_share": round(max(bal.values()) / total, 3) if total else 0.0,
               "top": max(bal, key=bal.get) if total else None}
        if prev and prev["issued_mP"] <= row["issued_mP"]:  # a new chain resets the totals
            row["issued_day_mP"] = row["issued_mP"] - prev["issued_mP"]
            row["burned_day_mP"] = row["burned_mP"] - prev["burned_mP"]
        out.append(row)
        prev = row
    return out


def velocity(fl: list[dict[str, Any]], clock: Clock, days: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Value moved between accounts per day, and how many times the circulating supply that is."""
    moved: Counter = Counter()
    for f in fl:
        moved[clock.day(f["height"])] += f["mP"]
    circ = {d["day"]: d["circulating_mP"] for d in days}
    return {day: {"moved_mP": v, "velocity": round(v / circ[day], 4) if circ.get(day) else None}
            for day, v in sorted(moved.items())}


def research_vs_acceptance(journal: sqlite3.Connection, board: Board, since: float | None) -> dict[str, Any]:
    """Answers given after searching, reading or buying information, against answers given
    without: how often each kind was accepted (05 doc 5: does research pay?)."""
    and_where, args = ("AND ts >= ?", (since,)) if since else ("", ())
    used: dict[tuple, set] = defaultdict(set)
    answered: dict[tuple, list] = defaultdict(list)
    for run, ep, node, tool, bid in journal.execute(
            "SELECT run_id, episode, node, json_extract(data, '$.tool'), json_extract(data, '$.args.bounty_id') "
            f"FROM entries WHERE kind = 'tool' AND episode IS NOT NULL {and_where}", args):
        key = (run, ep, node)
        if tool == "answer" and bid:
            answered[key].append(bid)
        else:
            used[key].add(tool)
    groups: dict[str, Counter] = {"조사 후": Counter(), "정보 구매 후": Counter(), "바로": Counter()}
    for key, bids in answered.items():
        tools = used.get(key, set())
        group = "정보 구매 후" if "offer_buy" in tools else "조사 후" if tools & set(RESEARCH_TOOLS) else "바로"
        for bid in bids:
            b = board.bounties.get(bid)
            if b is None or b.settled_at is None:
                continue
            groups[group]["answers"] += 1
            groups[group]["accepted"] += _accepted_author(board, b) == key[2]
            groups[group]["shared"] += str(b.outcome or b.state) == "AUTO_SPLIT"
    return {g: {"answers": c["answers"], "accepted": c["accepted"], "shared": c["shared"],
                "rate": round(c["accepted"] / c["answers"], 2) if c["answers"] else None}
            for g, c in groups.items()}


def market(board: Board, since_h: int = 0) -> dict[str, Any]:
    offers = [o for o in board.offers.values() if o.created_height >= since_h]
    purchases = [p.state for p in board.purchases.values() if p.height >= since_h]
    loans = [x.state for x in board.loans.values() if x.created_height >= since_h]
    return {"offers": len(offers), "purchases": dict(Counter(purchases)), "loans": dict(Counter(loans)),
            "licenses": len(board.licenses), "subscriptions": len(board.subscriptions)}


def _counts(by_state: dict[str, int]) -> str:
    if not by_state:
        return "0건"
    return f"{sum(by_state.values())}건(" + ", ".join(f"{k} {v}" for k, v in sorted(by_state.items())) + ")"


# ---- everything for one report ----

def analyse(board: Board, clock: Clock, journal: Path, knowledge: Path | None, since: float | None,
            base_personas: dict[str, str] | None = None) -> dict[str, Any]:
    since_h = clock.height_at(since) if since else 0
    fl = flows(board, since_h)
    j = sqlite3.connect(f"file:{journal}?mode=ro", uri=True)
    days = metrics_by_day(j, since)
    research = research_vs_acceptance(j, board, since)
    j.close()
    out: dict[str, Any] = {
        "since_height": since_h, "height": board.height,
        "pairs": pairs(board, fl, since_h), "competition": competition(board, since_h),
        "unattributed_mP": sum(f["mP"] for f in fl if f["payer"] is None),
        "days": days, "velocity": velocity(fl, clock, days), "research": research,
        "market": market(board, since_h),
    }
    if base_personas is None:
        from aichain.mastermind import prompt
        base_personas = {n: prompt.load(p) for n, p in PERSONAS.items()}
    rows: dict[str, list] = defaultdict(list)
    if knowledge is not None and knowledge.exists():
        k = sqlite3.connect(f"file:{knowledge}?mode=ro", uri=True)
        for node, ts, data in k.execute("SELECT node, ts, data FROM memories WHERE kind = 'persona' "
                                        "AND data IS NOT NULL AND data != '' ORDER BY node, seq"):
            rows[node].append({"ts": ts, "data": json.loads(data)})
        k.close()
    start = since or (clock.times[0] if clock.times else 0.0)
    end = clock.times[-1] if clock.times else time.time()
    points = [time.mktime(time.strptime(d["day"], "%Y-%m-%d")) + 86399 for d in days]
    out["persona"] = convergence(persona_timeline(base_personas, rows), start, end,
                                 [p for p in points if start <= p <= end + 86400])
    return out


# ---- Markdown ----

def _p(mp: int | None) -> str:
    return "-" if mp is None else f"{mp / 1000:,.1f}P"


def _acc(t: tuple[int, float | None, float | None]) -> str:
    n, here, elsewhere = t
    if not n:
        return "-"
    return f"{n}개 중 {here:.0%}" + (f" (다른 질문자 {elsewhere:.0%})" if elsewhere is not None else "")


def render_markdown(a: dict[str, Any]) -> str:
    out = ["## M7 분석", "",
           f"블록 #{a['since_height']} ~ #{a['height']}. 금액은 노드 사이에 실제로 오간 지급(현상금, 정보, 도구, 대출, 송금)이에요.", ""]
    c = a["competition"]
    out += ["### 답변 경쟁", "",
            f"정산된 질문 {c['bounties']}개 · 질문당 답한 노드 수: "
            + (", ".join(f"{k}명 {v}개" for k, v in c["answerers_per_bounty"].items()) or "-"), "",
            "| 노드 | 질문 | 답변 | 채택 |", "|---|---|---|---|"]
    out += [f"| {n} | {c['asked'][n]} | {c['answers'][n]} | {c['wins'][n]} |" for n in c["asked"]]
    out += ["", "### 담합 신호 (노드 쌍)", "",
            "| 쌍 | 지급 A→B / B→A | 상호성 | 수입 비중 B←A / A←B | A가 B 답 채택 | B가 A 답 채택 | 평점 A→B / B→A | 투표 일치 | 신호 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in a["pairs"]:
        x, y = r["pair"].split("-")
        out.append(f"| {r['pair']} | {_p(r['a_to_b_mP'])} / {_p(r['b_to_a_mP'])} | {r['reciprocity']} | "
                   f"{r['b_income_share_from_a']:.0%} / {r['a_income_share_from_b']:.0%} | "
                   f"{_acc(r[f'accept_{x}_of_{y}'])} | {_acc(r[f'accept_{y}_of_{x}'])} | "
                   f"{r['rating_a_to_b'] if r['rating_a_to_b'] is not None else '-'} / "
                   f"{r['rating_b_to_a'] if r['rating_b_to_a'] is not None else '-'} | "
                   f"{r['votes_agree'] if r['votes_agree'] is not None else '-'} | {', '.join(r['flags']) or '-'} |")
    flagged = [r for r in a["pairs"] if r["flags"]]
    out += ["", f"신호가 있는 쌍: {len(flagged)}개. 기준: 오간 금액 {MIN_VOLUME_mP // 1000}P 이상에서 상호성 "
                f"{RECIPROCITY_FLAG} 이상, 한 노드 수입의 {INCOME_SHARE_FLAG:.0%} 이상이 한 상대에게서 오고 그 상대의 "
                f"지출 비중보다 {EXCESS_SHARE_FLAG:.0%}p 이상 많음, "
                f"답 {MIN_ANSWERS}개 이상에서 채택률이 다른 질문자보다 {ACCEPT_BIAS_FLAG:.0%}p 이상 높음, 서로 평점 1.5 이상.", ""]
    if a["unattributed_mP"]:
        out += [f"지급자를 알 수 없는 수입 {_p(a['unattributed_mP'])} (규칙 2 이전 기록)", ""]
    p = a["persona"]
    out += ["### 성향 수렴", "",
            f"노드 성격 프롬프트의 평균 유사도: 기간 시작 {p['mean_start']} → 끝 {p['mean_end']} "
            f"({p['mean_end'] - p['mean_start']:+.4f}). 오르면 노드들이 비슷해지고 있다는 뜻이에요.", "",
            "| 쌍 | 시작 | 끝 |", "|---|---|---|"]
    out += [f"| {k} | {p['start'][k]} | {p['end'][k]} |" for k in p["start"]]
    out += ["", "기간 중 성향 수정과 변화량(1 − 자기 시작점과의 유사도): "
            + ", ".join(f"{n} {p['edits'][n]}회/{p['drift'][n]}" for n in sorted(p["edits"])), ""]
    days = a["days"]
    if days:
        first, last = days[0], days[-1]
        out += ["### 포인트 집중", "",
                f"지니 계수 {first['gini']} → {last['gini']}, 가장 많이 가진 노드의 몫 "
                f"{first['top_share']:.0%}({first['top']}) → {last['top_share']:.0%}({last['top']}).", "",
                "### 발행·소각·유통 속도 (하루 단위)", "",
                "| 날짜 | 발행 | 소각 | 노드 간 이동 | 유통 속도 | 지니 |", "|---|---|---|---|---|---|"]
        for d in days:
            v = a["velocity"].get(d["day"], {})
            out.append(f"| {d['day']} | {_p(d.get('issued_day_mP'))} | {_p(d.get('burned_day_mP'))} | "
                       f"{_p(v.get('moved_mP', 0))} | {v.get('velocity') if v.get('velocity') is not None else '-'} | "
                       f"{d['gini']} |")
        out.append("")
    out += ["### 조사와 채택", "", "| 답하기 전 | 답변 | 채택 | 채택률 | 자동 분배 |", "|---|---|---|---|---|"]
    for g, r in a["research"].items():
        rate = "-" if r["rate"] is None else f"{r['rate']:.0%}"
        out.append(f"| {g} | {r['answers']} | {r['accepted']} | {rate} | {r['shared']} |")
    m = a["market"]
    out += ["", f"시장: 판매 등록 {m['offers']}건, 구매 {_counts(m['purchases'])}, 대출 {_counts(m['loans'])}, "
                f"도구 라이선스 {m['licenses']}건, 구독 {m['subscriptions']}건", ""]
    return "\n".join(out)
