import difflib
import re


def shingles(text, n):
    toks = re.findall(r"[\w가-힣]+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0))} or set(toks)


def run(input, ctx):
    a, b = str(input.get("a", "")), str(input.get("b", ""))
    n = int(input.get("shingle", 3))
    sa, sb = shingles(a, n), shingles(b, n)
    jaccard = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    ratio = matcher.ratio()
    verdict = "near-copy" if jaccard > 0.6 or ratio > 0.85 else "similar" if jaccard > 0.3 or ratio > 0.6 else "different"
    return {"jaccard": round(jaccard, 3), "sequence_ratio": round(ratio, 3),
            "longest_common": a[match.a:match.a + match.size][:300], "longest_common_len": match.size,
            "verdict": verdict}
