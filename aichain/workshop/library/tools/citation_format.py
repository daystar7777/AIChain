import re


def run(input, ctx):
    style = input.get("style", "markdown")
    out = []
    for i, s in enumerate(input.get("sources", []), 1):
        url = str(s.get("url", ""))
        title = str(s.get("title") or url)
        host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        date = str(s.get("date", ""))[:10]
        if style == "markdown":
            out.append(f"{i}. [{title}]({url}) — {host}" + (f", {date}" if date else ""))
        else:
            out.append(f"[{i}] {title}. {host}." + (f" {date}." if date else "") + f" {url}")
    return {"text": "\n".join(out), "count": len(out)}
