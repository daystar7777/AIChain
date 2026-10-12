import datetime

WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def parse(s):
    return datetime.date.fromisoformat(str(s)[:10])


def run(input, ctx):
    op = input.get("op", "diff")
    try:
        a = parse(input["date"])
        if op == "diff":
            b = parse(input["other"])
            days = (b - a).days
            return {"ok": True, "days": days, "weeks": round(days / 7, 3),
                    "business_days": sum(1 for i in range(abs(days)) if (min(a, b) + datetime.timedelta(i)).weekday() < 5)}
        if op == "add":
            r = a + datetime.timedelta(days=int(input.get("days", 0)))
            return {"ok": True, "date": r.isoformat(), "weekday": WEEKDAYS[r.weekday()]}
        if op == "info":
            return {"ok": True, "weekday": WEEKDAYS[a.weekday()], "iso_week": a.isocalendar()[1],
                    "day_of_year": a.timetuple().tm_yday, "quarter": (a.month - 1) // 3 + 1}
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "op must be diff, add or info"}
