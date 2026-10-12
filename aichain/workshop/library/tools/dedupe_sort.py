def run(input, ctx):
    items = input.get("items", [])
    key = input.get("key")
    seen, unique = set(), []
    for it in items:
        k = str(it.get(key)) if key and isinstance(it, dict) else str(it).strip().lower() if input.get("ignore_case") else str(it)
        if k not in seen:
            seen.add(k)
            unique.append(it)
    out = {"unique": unique, "removed": len(items) - len(unique)}
    if input.get("sort"):
        try:
            out["sorted"] = sorted(unique, key=lambda x: x.get(key) if key and isinstance(x, dict) else x,
                                   reverse=bool(input.get("reverse")))
        except TypeError:
            out["sorted"] = sorted(unique, key=str, reverse=bool(input.get("reverse")))
    return out
