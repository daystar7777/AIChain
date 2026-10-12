import re

STOP = {"그리고", "또는", "것", "수", "등", "및", "를", "을", "이", "가", "은", "는", "의", "에", "로", "으로",
        "the", "a", "an", "and", "or", "of", "to", "in", "with", "for", "is", "be", "that", "it"}


def terms(text):
    out = []
    for w in re.findall(r"[A-Za-z_][\w.()]*|[가-힣]{2,}|\d+", text):
        w = re.sub(r"(을|를|이|가|은|는|의|에|로|으로|과|와|도|만)$", "", w) if re.match(r"[가-힣]", w) else w
        if len(w) > 1 and w.lower() not in STOP and w not in out:
            out.append(w)
    return out


def run(input, ctx):
    answer, criteria = str(input.get("answer", "")), str(input.get("criteria", ""))
    keys = terms(criteria)
    low = answer.lower()
    found = [k for k in keys if k.lower() in low]
    missing = [k for k in keys if k.lower() not in low]
    has_code = "```" in answer or bool(re.search(r"^\s{4}\S", answer, re.M))
    wants_code = bool(re.search(r"코드|예시|example|code|실행", criteria, re.I))
    return {"coverage": round(len(found) / len(keys), 3) if keys else 1.0, "found": found,
            "missing": missing, "has_code": has_code,
            "code_required_but_missing": wants_code and not has_code,
            "answer_chars": len(answer)}
