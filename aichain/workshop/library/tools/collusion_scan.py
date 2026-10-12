import collections


def run(input, ctx):
    """trades: [{asker, answerer, accepted(bool), height, reward}] -> pair concentration and flags."""
    trades = input.get("trades", [])
    min_pairs = int(input.get("min_pair_trades", 3))
    pairs = collections.Counter()
    accepted = collections.Counter()
    paid = collections.Counter()
    income = collections.Counter()
    for t in trades:
        a, b = t.get("asker"), t.get("answerer")
        if not a or not b:
            continue
        pairs[(a, b)] += 1
        if t.get("accepted"):
            accepted[(a, b)] += 1
            paid[(a, b)] += float(t.get("reward", 0))
            income[b] += float(t.get("reward", 0))
    flags, rows = [], []
    for (a, b), n in pairs.most_common():
        back = accepted.get((b, a), 0)
        share = paid[(a, b)] / income[b] if income[b] else 0.0
        rate = accepted[(a, b)] / n
        row = {"asker": a, "answerer": b, "trades": n, "accept_rate": round(rate, 3),
               "reciprocal_accepts": back, "share_of_answerer_income": round(share, 3)}
        reasons = []
        if n >= min_pairs and rate >= 0.9:
            reasons.append("almost always accepts this answerer")
        if back >= 1 and accepted[(a, b)] >= 1 and n >= min_pairs:
            reasons.append("reciprocal accepting")
        if share >= 0.7 and income[b] > 0 and n >= min_pairs:
            reasons.append("answerer's income comes mostly from this asker")
        if reasons:
            row["reasons"] = reasons
            flags.append(row)
        rows.append(row)
    return {"pairs": rows[:50], "flags": flags, "trades": len(trades),
            "note": "flags are leads, not proof: check answer quality before disputing"}
