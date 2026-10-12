import ast


def run(input, ctx):
    code = str(input.get("code", ""))
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "error": e.msg, "line": e.lineno, "offset": e.offset,
                "text": (e.text or "").strip()[:200]}
    funcs, classes, imports = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return {"ok": True, "functions": funcs, "classes": classes, "imports": sorted(set(imports)),
            "lines": len(code.splitlines())}
