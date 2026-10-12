import ast

BRANCHES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp, ast.IfExp, ast.comprehension,
            ast.ExceptHandler, ast.Match)


def run(input, ctx):
    code = str(input.get("code", ""))
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"line {e.lineno}: {e.msg}"}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = 1 + sum(isinstance(n, BRANCHES) for n in ast.walk(node))
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            out.append({"name": node.name, "complexity": complexity, "lines": length,
                        "args": len(node.args.args)})
    out.sort(key=lambda f: -f["complexity"])
    code_lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    return {"ok": True, "functions": out, "code_lines": len(code_lines),
            "max_complexity": out[0]["complexity"] if out else 0,
            "hotspots": [f["name"] for f in out if f["complexity"] > 10]}
