import base64
import hashlib
import zlib


def run(input, ctx):
    op = input.get("op", "sha256")
    text = str(input.get("text", ""))
    data = text.encode("utf-8")
    try:
        if op in ("sha256", "sha1", "md5", "sha512"):
            return {"ok": True, "result": hashlib.new(op, data).hexdigest()}
        if op == "crc32":
            return {"ok": True, "result": format(zlib.crc32(data) & 0xFFFFFFFF, "08x")}
        if op == "base64_encode":
            return {"ok": True, "result": base64.b64encode(data).decode()}
        if op == "base64_decode":
            return {"ok": True, "result": base64.b64decode(text, validate=True).decode("utf-8", "replace")}
        if op == "hex_encode":
            return {"ok": True, "result": data.hex()}
    except (ValueError, base64.binascii.Error) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "op must be sha256, sha1, md5, sha512, crc32, base64_encode, base64_decode, hex_encode"}
