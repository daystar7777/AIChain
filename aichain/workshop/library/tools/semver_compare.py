import re


def parse(v):
    m = re.match(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-.]?([0-9A-Za-z.-]+))?", str(v).strip())
    if not m:
        raise ValueError(f"not a version: {v}")
    major, minor, patch, pre = m.groups()
    return int(major), int(minor or 0), int(patch or 0), pre or ""


def run(input, ctx):
    try:
        a, b = parse(input.get("a")), parse(input.get("b"))
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    ka = a[:3] + ((1,) if not a[3] else (0,)) + (a[3],)
    kb = b[:3] + ((1,) if not b[3] else (0,)) + (b[3],)
    cmp = (ka > kb) - (ka < kb)
    bump = "major" if a[0] != b[0] else "minor" if a[1] != b[1] else "patch" if a[2] != b[2] else "pre" if a[3] != b[3] else "same"
    return {"ok": True, "compare": cmp, "newer": "a" if cmp > 0 else "b" if cmp < 0 else "equal", "difference": bump}
