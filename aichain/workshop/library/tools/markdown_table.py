def cell(v):
    return str(v).replace("|", "\\|").replace("\n", " ")


def run(input, ctx):
    rows = input.get("rows", [])
    if not rows:
        return {"markdown": ""}
    if isinstance(rows[0], dict):
        cols = input.get("columns") or list(dict.fromkeys(k for r in rows for k in r))
        body = [[cell(r.get(c, "")) for c in cols] for r in rows]
    else:
        cols = input.get("columns") or [f"c{i + 1}" for i in range(max(len(r) for r in rows))]
        body = [[cell(x) for x in r] + [""] * (len(cols) - len(r)) for r in rows]
    lines = ["| " + " | ".join(cell(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return {"markdown": "\n".join(lines)}
