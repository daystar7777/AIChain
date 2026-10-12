import collections
import re

STOP = {"그리고", "그러나", "하지만", "또는", "있다", "없다", "한다", "하는", "이다", "것이", "것을", "수", "등",
        "the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "have", "has", "not",
        "but", "you", "your", "its", "can", "will", "into", "than", "then", "they", "their", "about"}


def run(input, ctx):
    text = str(input.get("text", ""))
    k = int(input.get("k", 10))
    words = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}|[가-힣]{2,}", text):
        w = re.sub(r"(을|를|이|가|은|는|의|에|에서|으로|로|과|와|도|만|들)$", "", w) if re.match(r"[가-힣]", w) else w.lower()
        if len(w) > 1 and w not in STOP:
            words.append(w)
    counts = collections.Counter(words)
    return {"keywords": [{"term": t, "count": c} for t, c in counts.most_common(k)], "unique_terms": len(counts)}
