import math

W = {"A": 0.30, "V": 0.15, "J": 0.15, "T": 0.10, "W": 0.10, "U": 0.10, "S": 0.10}


def run(input, ctx):
    """What-if reputation per the 03 doc formula (floats; the chain uses integer fixed point)."""
    acc, auto = float(input.get("accepted", 0)), float(input.get("auto", 0))
    rej, lost = float(input.get("rejected", 0)), float(input.get("lost_answer_disputes", 0))
    income = float(input.get("income_P", 0))
    created, timeouts = float(input.get("bounties_created", 0)), float(input.get("judge_timeouts", 0))
    sold, lost_tool = float(input.get("tools_sold", 0)), float(input.get("lost_tool_disputes", 0))
    won, lost_d = float(input.get("disputes_won", 0)), float(input.get("disputes_lost", 0))
    epochs = float(input.get("heartbeat_epochs", 0))
    ratings = input.get("ratings", [])
    terms = {
        "A": (acc + 0.5 * auto + 1) / (acc + auto + rej + 3 * lost + 2),
        "V": min(1.0, math.log10(1 + income) / 3),
        "J": 1 - timeouts / (created + 2),
        "T": (1 + sold) / (2 + sold + 3 * lost_tool) if sold else 0.5,
        "W": (won + 1) / (won + lost_d + 2),
        "U": min(1.0, epochs / 720),
        "S": 0.5 + 0.5 * sum(r.get("score", 0) * r.get("rater_rep", 0.45) for r in ratings)
        / (sum(2 * r.get("rater_rep", 0.45) for r in ratings) or 1) if ratings else 0.5,
    }
    value = sum(W[k] * v for k, v in terms.items())
    weakest = max(terms, key=lambda k: W[k] * (1 - terms[k]))
    gains = {k: round(W[k] * (1 - v), 4) for k, v in terms.items()}
    return {"reputation": round(value, 4), "terms": {k: round(v, 4) for k, v in terms.items()},
            "max_gain_by_term": dict(sorted(gains.items(), key=lambda x: -x[1])), "weakest_term": weakest}
