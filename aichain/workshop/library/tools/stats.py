import statistics


def pct(xs, p):
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(input, ctx):
    xs = sorted(float(x) for x in input.get("values", []))
    if not xs:
        return {"count": 0}
    out = {"count": len(xs), "sum": sum(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
           "min": xs[0], "max": xs[-1], "p25": pct(xs, 25), "p75": pct(xs, 75), "p90": pct(xs, 90)}
    if len(xs) > 1:
        out["stdev"] = statistics.stdev(xs)
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in out.items()}
