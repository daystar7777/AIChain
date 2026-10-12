"""Encrypted backup and restore of a node host (for moving to new hardware).

    python -m aichain.backup create --out aichain-backup.enc     # passphrase asked (or AICHAIN_BACKUP_PASSPHRASE)
    python -m aichain.backup restore aichain-backup.enc --to /opt/aichain
    python -m aichain.backup verify aichain-backup.enc

Bundles everything a node's identity and state live in: keys (data/chain/keys), the signed chain,
memories and knowledge, workshop tools, approvals, usage counters, node settings and tokens,
journal and reports. The archive is encrypted with a key derived from the passphrase (Argon2id +
XSalsa20-Poly1305). Stop the services before creating a backup used for a move (see
docs/notes/node-migration.md) so the old host does not keep signing.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from nacl import pwhash, secret, utils

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"AICHAINBK1"
INCLUDE = ["data", ".env"]
EXCLUDE_SUFFIXES = (".tmp", "-shm")


def _key(passphrase: str, salt: bytes) -> bytes:
    return pwhash.argon2id.kdf(secret.SecretBox.KEY_SIZE, passphrase.encode(), salt,
                               opslimit=pwhash.argon2id.OPSLIMIT_MODERATE,
                               memlimit=pwhash.argon2id.MEMLIMIT_MODERATE)


def create(src: Path, passphrase: str) -> bytes:
    buf = io.BytesIO()
    manifest = {"created": time.time(), "root": str(src), "files": []}
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            path = src / name
            if not path.exists():
                continue
            for f in ([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())):
                if f.name.endswith(EXCLUDE_SUFFIXES):
                    continue
                arc = str(f.relative_to(src))
                tar.add(f, arcname=arc)
                manifest["files"].append(arc)
        info = tarfile.TarInfo("BACKUP_MANIFEST.json")
        raw = json.dumps(manifest).encode()
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    salt = utils.random(pwhash.argon2id.SALTBYTES)
    box = secret.SecretBox(_key(passphrase, salt))
    return MAGIC + salt + box.encrypt(buf.getvalue())


def open_backup(blob: bytes, passphrase: str) -> tarfile.TarFile:
    if not blob.startswith(MAGIC):
        raise ValueError("not an AIChain backup")
    salt = blob[len(MAGIC):len(MAGIC) + pwhash.argon2id.SALTBYTES]
    data = secret.SecretBox(_key(passphrase, salt)).decrypt(blob[len(MAGIC) + len(salt):])
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def restore(blob: bytes, passphrase: str, dest: Path) -> list[str]:
    with open_backup(blob, passphrase) as tar:
        members = [m for m in tar.getmembers() if m.name != "BACKUP_MANIFEST.json"]
        for m in members:  # refuse path tricks
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest.resolve())) or m.issym() or m.islnk():
                raise ValueError(f"unsafe path in backup: {m.name}")
        tar.extractall(dest, members=members, filter="data")
    for sensitive in ("data/chain/keys", ".env", "data/admin_token", "data/bridge_owner_token"):
        p = dest / sensitive
        if p.is_dir():
            os.chmod(p, 0o700)
            for f in p.iterdir():
                os.chmod(f, 0o600)
        elif p.exists():
            os.chmod(p, 0o600)
    return [m.name for m in members]


def _passphrase(confirm: bool) -> str:
    env = os.environ.get("AICHAIN_BACKUP_PASSPHRASE")
    if env:
        return env
    p = getpass.getpass("backup passphrase: ")
    if confirm and p != getpass.getpass("again: "):
        raise SystemExit("passphrases differ")
    if len(p) < 12:
        raise SystemExit("use at least 12 characters")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Encrypted backup / restore of an AIChain host")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--src", type=Path, default=ROOT)
    r = sub.add_parser("restore")
    r.add_argument("file", type=Path)
    r.add_argument("--to", type=Path, default=ROOT)
    v = sub.add_parser("verify")
    v.add_argument("file", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "create":
        blob = create(args.src, _passphrase(confirm=True))
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        print(f"wrote {args.out} ({len(blob) / 1e6:.1f} MB)")
    elif args.cmd == "verify":
        with open_backup(args.file.read_bytes(), _passphrase(confirm=False)) as tar:
            manifest = json.load(tar.extractfile("BACKUP_MANIFEST.json"))
        print(f"ok: {len(manifest['files'])} files, created {time.ctime(manifest['created'])}")
    else:
        names = restore(args.file.read_bytes(), _passphrase(confirm=False), args.to)
        print(f"restored {len(names)} files into {args.to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
