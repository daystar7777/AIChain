def run(input, ctx):
    value = str(input.get("value", "")).strip().lower()
    src = int(input.get("from_base", 10))
    try:
        n = int(value, src)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"

    def to(base):
        if n == 0:
            return "0"
        x, out = abs(n), ""
        while x:
            x, r = divmod(x, base)
            out = digits[r] + out
        return ("-" if n < 0 else "") + out

    targets = input.get("to_bases") or [2, 8, 10, 16]
    return {"ok": True, "decimal": n, "results": {str(b): to(int(b)) for b in targets if 2 <= int(b) <= 36}}
