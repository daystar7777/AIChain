import re


def run(input, ctx):
    pattern, text = str(input.get("pattern", "")), str(input.get("text", ""))
    flags = 0
    for f in str(input.get("flags", "")):
        flags |= {"i": re.I, "m": re.M, "s": re.S, "x": re.X}.get(f, 0)
    if len(pattern) > 500 or len(text) > 50000:
        return {"ok": False, "error": "pattern or text too long"}
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return {"ok": False, "error": f"invalid pattern: {e}"}
    matches = []
    for m in rx.finditer(text):
        matches.append({"match": m.group(0)[:200], "start": m.start(), "groups": [g for g in m.groups()][:10]})
        if len(matches) >= 50:
            break
    return {"ok": True, "count": len(matches), "matches": matches}
