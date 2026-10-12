import difflib


def run(input, ctx):
    a = str(input.get("a", "")).splitlines()
    b = str(input.get("b", "")).splitlines()
    diff = list(difflib.unified_diff(a, b, "a", "b", lineterm="", n=int(input.get("context", 1))))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return {"diff": "\n".join(diff[:400]), "added": added, "removed": removed, "identical": not diff}
