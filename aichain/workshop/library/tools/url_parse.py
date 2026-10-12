import re


def unquote(s):
    out = re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), s.replace("+", " "))
    try:
        return out.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return out


def run(input, ctx):
    url = str(input.get("url", "")).strip()
    m = re.match(r"^(?:(?P<scheme>[a-zA-Z][\w+.-]*):)?(?://(?P<netloc>[^/?#]*))?(?P<path>[^?#]*)"
                 r"(?:\?(?P<query>[^#]*))?(?:#(?P<fragment>.*))?$", url)
    if not m:
        return {"ok": False}
    netloc = m.group("netloc") or ""
    host = netloc.rsplit("@", 1)[-1]
    port = None
    if re.search(r":\d+$", host):
        host, port = host.rsplit(":", 1)
    params = {}
    for pair in (m.group("query") or "").split("&"):
        if pair:
            k, _, v = pair.partition("=")
            params.setdefault(unquote(k), []).append(unquote(v))
    parts = host.lower().split(".")
    return {"ok": True, "scheme": (m.group("scheme") or "").lower(), "host": host.lower(),
            "port": int(port) if port else None,
            "domain": ".".join(parts[-2:]) if len(parts) >= 2 else host.lower(),
            "path": unquote(m.group("path") or ""), "query": params, "fragment": m.group("fragment") or ""}
