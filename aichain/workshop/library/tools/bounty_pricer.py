def run(input, ctx):
    """Suggest a reward within the 10% per-bounty cap of the available balance."""
    balance = float(input.get("available_P", 0))
    difficulty = max(1, min(5, int(input.get("difficulty", 2))))
    urgency = max(1, min(3, int(input.get("urgency", 1))))
    needs_research = bool(input.get("needs_research", False))
    cap = balance * 0.10
    base = {1: 2, 2: 5, 3: 10, 4: 20, 5: 40}[difficulty]
    suggested = base * (1 + 0.25 * (urgency - 1)) * (1.3 if needs_research else 1.0)
    suggested = max(1.0, min(suggested, cap))
    return {"suggested_P": round(suggested, 1), "cap_P": round(cap, 1),
            "capped": suggested >= cap, "reasoning": f"difficulty {difficulty}, urgency {urgency}"
            + (", research" if needs_research else "")}
