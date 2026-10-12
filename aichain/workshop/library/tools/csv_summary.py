import csv
import statistics


def num(v):
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def run(input, ctx):
    lines = str(input.get("csv", "")).splitlines()
    rows = list(csv.reader(lines, delimiter=str(input.get("delimiter", ","))))
    if not rows:
        return {"rows": 0, "columns": []}
    header, data = rows[0], rows[1:]
    cols = []
    for i, name in enumerate(header):
        values = [r[i] for r in data if i < len(r)]
        nums = [x for x in (num(v) for v in values) if x is not None]
        col = {"name": name, "non_empty": sum(1 for v in values if v.strip()), "unique": len(set(values))}
        if nums and len(nums) >= len(values) * 0.8:
            col.update({"type": "number", "min": min(nums), "max": max(nums),
                        "mean": round(statistics.fmean(nums), 6)})
        else:
            col.update({"type": "text", "examples": list(dict.fromkeys(values))[:5]})
        cols.append(col)
    return {"rows": len(data), "columns": cols}
