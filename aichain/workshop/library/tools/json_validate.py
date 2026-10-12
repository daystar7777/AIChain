TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float),
         "boolean": bool, "null": type(None)}  # noqa


def tname(value):
    for name, t in TYPES.items():
        if isinstance(value, t) and not (name in ("integer", "number") and isinstance(value, bool)):
            return name
    return "unknown"


def check(value, schema, path, errors):
    t = schema.get("type")
    if t:
        allowed = t if isinstance(t, list) else [t]
        ok = any(isinstance(value, TYPES[x]) and not (x in ("integer", "number") and isinstance(value, bool))
                 for x in allowed if x in TYPES)
        if not ok:
            errors.append(f"{path}: expected {t}, got {tname(value)}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required '{key}'")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                check(value[key], sub, f"{path}.{key}", errors)
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected property '{key}'")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            check(item, schema["items"], f"{path}[{i}]", errors)


def run(input, ctx):
    errors = []
    check(input.get("data"), input.get("schema") or {}, "$", errors)
    return {"valid": not errors, "errors": errors[:50]}
