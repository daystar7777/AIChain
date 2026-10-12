import re


def run(input, ctx):
    text = str(input.get("text", ""))
    words = re.findall(r"[\w가-힣]+", text)
    sentences = [s for s in re.split(r"(?<=[.!?。])\s+|\n+", text) if s.strip()]
    hangul = len(re.findall(r"[가-힣]", text))
    letters = len(re.findall(r"[^\W\d_]", text))
    return {
        "chars": len(text),
        "chars_no_space": len(re.sub(r"\s", "", text)),
        "words": len(words),
        "lines": len(text.splitlines()) if text else 0,
        "sentences": len(sentences),
        "hangul_ratio": round(hangul / letters, 3) if letters else 0.0,
        "avg_word_len": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
    }
