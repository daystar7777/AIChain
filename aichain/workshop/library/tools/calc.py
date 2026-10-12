import ast
import math
import operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
       ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
         "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round, "floor": math.floor,
         "ceil": math.ceil, "min": min, "max": max, "factorial": math.factorial, "comb": math.comb,
         "perm": math.perm, "gcd": math.gcd}
CONSTS = {"pi": math.pi, "e": math.e}


def ev(node):
    if isinstance(node, ast.Expression):
        return ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = ev(node.left), ev(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](ev(node.operand))
    if isinstance(node, ast.Name) and node.id in CONSTS:
        return CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS:
        return FUNCS[node.func.id](*[ev(a) for a in node.args])
    raise ValueError("unsupported expression element")


def run(input, ctx):
    expr = str(input.get("expression", ""))
    if len(expr) > 500:
        return {"ok": False, "error": "expression too long"}
    try:
        value = ev(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "value": value}
