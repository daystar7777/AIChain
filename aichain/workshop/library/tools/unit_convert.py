UNITS = {
    "length": {"m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144},
    "mass": {"kg": 1, "g": 0.001, "mg": 1e-6, "t": 1000, "lb": 0.45359237, "oz": 0.028349523125},
    "time": {"s": 1, "ms": 0.001, "min": 60, "h": 3600, "day": 86400, "week": 604800},
    "data": {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3},
    "speed": {"m/s": 1, "km/h": 1 / 3.6, "mph": 0.44704, "knot": 0.514444},
}


def temp(value, src, dst):
    c = {"C": value, "F": (value - 32) * 5 / 9, "K": value - 273.15}[src]
    return {"C": c, "F": c * 9 / 5 + 32, "K": c + 273.15}[dst]


def run(input, ctx):
    value, src, dst = float(input.get("value", 0)), str(input.get("from", "")), str(input.get("to", ""))
    if src in ("C", "F", "K") and dst in ("C", "F", "K"):
        return {"ok": True, "value": round(temp(value, src, dst), 6), "kind": "temperature"}
    for kind, table in UNITS.items():
        if src in table and dst in table:
            return {"ok": True, "value": round(value * table[src] / table[dst], 9), "kind": kind}
    return {"ok": False, "error": f"cannot convert {src} to {dst}",
            "known": {k: sorted(v) for k, v in UNITS.items()} | {"temperature": ["C", "F", "K"]}}
