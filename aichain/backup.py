"""Encrypted backup and restore of a node host (for moving to new hardware, and daily).

    python -m aichain.backup create --out aichain-backup.enc     # passphrase asked (or AICHAIN_BACKUP_PASSPHRASE)
    python -m aichain.backup restore aichain-backup.enc --to /opt/aichain
    python -m aichain.backup verify aichain-backup.enc

Public-key mode, for unattended backups (the server never holds anything that can decrypt):

    python -m aichain.backup keygen --out ~/aichain-backup.key   # on the operator's own computer
    python -m aichain.backup create --to-pubkey <public key> --out aichain-backup.enc
    python -m aichain.backup auto --dir /opt/aichain/backups --keep 7   # daily timer; key from .env
    python -m aichain.backup verify|restore aichain-backup.enc --key-file ~/aichain-backup.key

Bundles everything a node's identity and state live in: keys (data/chain/keys), the signed chain,
memories and knowledge, workshop tools, approvals, usage counters, node settings and tokens,
journal and reports. SQLite databases are copied through SQLite's online backup, so a backup
taken while the services run is consistent. Passphrase mode: Argon2id + XSalsa20-Poly1305 over
the whole archive. Public-key mode: a random key sealed to the operator's X25519 public key
(SealedBox) and the archive streamed through XChaCha20-Poly1305 secretstream in 64 KiB chunks,
so memory stays small however big the data is. Stop the services before creating a backup used
for a move (see docs/notes/node-migration.md) so the old host does not keep signing.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import getpass
import io
import json
import os
import sqlite3
import struct
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from nacl import bindings as sodium
from nacl import pwhash, secret, utils
from nacl.public import PrivateKey, PublicKey, SealedBox

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"AICHAINBK1"  # passphrase
MAGIC_PUB = b"AICHAINBK2"  # public key, streamed
INCLUDE = ["data", ".env"]
EXCLUDE_SUFFIXES = (".tmp", "-shm")
# Daily backups leave out what is large and only observational (the journal of every LLM call,
# the trace files); keys, chain, knowledge, tools and settings are all kept.
AUTO_EXCLUDE = ["data/journal", "data/bridge-*.jsonl", "data/trace-*.jsonl"]
CHUNK = 64 * 1024
_SEALED_KEY = sodium.crypto_secretstream_xchacha20poly1305_KEYBYTES + 48  # + SealedBox overhead
_HEADER = sodium.crypto_secretstream_xchacha20poly1305_HEADERBYTES
_TAG_MESSAGE = sodium.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE
_TAG_FINAL = sodium.crypto_secretstream_xchacha20poly1305_TAG_FINAL


def _key(passphrase: str, salt: bytes) -> bytes:
    return pwhash.argon2id.kdf(secret.SecretBox.KEY_SIZE, passphrase.encode(), salt,
                               opslimit=pwhash.argon2id.OPSLIMIT_MODERATE,
                               memlimit=pwhash.argon2id.MEMLIMIT_MODERATE)


# ---- the archive ----

def _excluded(arc: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(arc, p) or fnmatch.fnmatch(arc, p.rstrip("/") + "/*") for p in patterns)


def _sqlite_copy(path: Path, tmpdir: Path) -> Path | None:
    """A consistent copy of a live SQLite database (online backup API), or None if `path` is
    not one. Writers keep going meanwhile (WAL)."""
    with path.open("rb") as f:
        if f.read(16) != b"SQLite format 3\0":
            return None
    copy = Path(tempfile.mkstemp(dir=tmpdir, suffix=".db")[1])
    src = sqlite3.connect(path, timeout=60)
    try:
        dst = sqlite3.connect(copy)
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()
    return copy


def write_archive(src: Path, fileobj: BinaryIO, exclude: list[str] | None = None,
                  tmpdir: Path | None = None) -> dict:
    """Write the tar.gz of `src` to fileobj, sequentially (it need not be seekable)."""
    manifest: dict = {"created": time.time(), "root": str(src), "files": [], "excluded": exclude or []}
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmp, \
            tarfile.open(fileobj=fileobj, mode="w|gz") as tar:
        for name in INCLUDE:
            path = src / name
            if not path.exists():
                continue
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            snapshots: set[str] = set()
            for f in files:
                arc = str(f.relative_to(src))
                if f.name.endswith(EXCLUDE_SUFFIXES) or _excluded(arc, exclude or []):
                    continue
                if any(arc == s + suffix for s in snapshots for suffix in ("-wal", "-journal")):
                    continue  # already inside the consistent copy
                copy = _sqlite_copy(f, Path(tmp)) if f.suffix == ".db" else None
                if copy is not None:
                    snapshots.add(arc)
                    info = tar.gettarinfo(str(f), arcname=arc)
                    info.size = copy.stat().st_size
                    with copy.open("rb") as fh:
                        tar.addfile(info, fh)
                    copy.unlink()
                else:
                    try:
                        tar.add(f, arcname=arc)
                    except FileNotFoundError:  # removed meanwhile (rotated, replaced)
                        continue
                manifest["files"].append(arc)
        info = tarfile.TarInfo("BACKUP_MANIFEST.json")
        raw = json.dumps(manifest).encode()
        info.size = len(raw)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(raw))
    return manifest


# ---- passphrase mode (whole archive in memory; moving a host by hand) ----

def create(src: Path, passphrase: str, exclude: list[str] | None = None) -> bytes:
    buf = io.BytesIO()
    write_archive(src, buf, exclude)
    salt = utils.random(pwhash.argon2id.SALTBYTES)
    box = secret.SecretBox(_key(passphrase, salt))
    return MAGIC + salt + box.encrypt(buf.getvalue())


def open_backup(blob: bytes, passphrase: str) -> tarfile.TarFile:
    if not blob.startswith(MAGIC):
        raise ValueError("not an AIChain passphrase backup")
    salt = blob[len(MAGIC):len(MAGIC) + pwhash.argon2id.SALTBYTES]
    data = secret.SecretBox(_key(passphrase, salt)).decrypt(blob[len(MAGIC) + len(salt):])
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


# ---- public-key mode (streamed; unattended daily backups) ----

def keygen() -> tuple[str, str]:
    """(public, private) X25519 keys, base64."""
    sk = PrivateKey.generate()
    return (base64.b64encode(bytes(sk.public_key)).decode(), base64.b64encode(bytes(sk)).decode())


def _decode_key(text: str, kind: str) -> bytes:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except ValueError:
        raw = b""
    if len(raw) != 32:
        raise ValueError(f"not a base64 X25519 {kind} key (32 bytes)")
    return raw


def read_key_file(path: Path) -> bytes:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"no key in {path}")
    return _decode_key(lines[-1], "private")


class _Encryptor:
    """Write-only stream for tarfile: buffers CHUNK bytes and writes each as one secretstream message."""

    def __init__(self, out: BinaryIO, pubkey: bytes) -> None:
        key = sodium.crypto_secretstream_xchacha20poly1305_keygen()
        self._state = sodium.crypto_secretstream_xchacha20poly1305_state()
        header = sodium.crypto_secretstream_xchacha20poly1305_init_push(self._state, key)
        out.write(MAGIC_PUB + SealedBox(PublicKey(pubkey)).encrypt(key) + header)
        self._out = out
        self._buf = bytearray()

    def write(self, data: bytes) -> int:
        self._buf += data
        while len(self._buf) > CHUNK:
            self._push(bytes(self._buf[:CHUNK]), _TAG_MESSAGE)
            del self._buf[:CHUNK]
        return len(data)

    def _push(self, chunk: bytes, tag: int) -> None:
        c = sodium.crypto_secretstream_xchacha20poly1305_push(self._state, chunk, None, tag)
        self._out.write(struct.pack(">I", len(c)) + c)

    def finish(self) -> None:
        self._push(bytes(self._buf), _TAG_FINAL)  # authenticates the end: truncation is detected
        self._buf.clear()


def create_to_pubkey(src: Path, pubkey: bytes, out: BinaryIO, exclude: list[str] | None = None,
                     tmpdir: Path | None = None) -> dict:
    enc = _Encryptor(out, pubkey)
    manifest = write_archive(src, enc, exclude, tmpdir)
    enc.finish()
    return manifest


def decrypt_stream(inp: BinaryIO, private_key: bytes, out: BinaryIO) -> None:
    """Public-key backup -> the plain tar.gz. Raises on a wrong key, tampering or truncation."""
    if inp.read(len(MAGIC_PUB)) != MAGIC_PUB:
        raise ValueError("not an AIChain public-key backup")
    key = SealedBox(PrivateKey(private_key)).decrypt(inp.read(_SEALED_KEY))
    state = sodium.crypto_secretstream_xchacha20poly1305_state()
    sodium.crypto_secretstream_xchacha20poly1305_init_pull(state, inp.read(_HEADER), key)
    while True:
        size = inp.read(4)
        if len(size) < 4:
            raise ValueError("backup is truncated")
        chunk, tag = sodium.crypto_secretstream_xchacha20poly1305_pull(
            state, inp.read(struct.unpack(">I", size)[0]), None)
        out.write(chunk)
        if tag == _TAG_FINAL:
            if inp.read(1):
                raise ValueError("unexpected data after the end of the backup")
            return


@contextmanager
def open_backup_file(path: Path, passphrase: str | None = None,
                     private_key: bytes | None = None) -> Iterator[tarfile.TarFile]:
    """Either format. Public-key backups are decrypted to a temporary file, not into memory."""
    with path.open("rb") as f:
        magic = f.read(len(MAGIC_PUB))
        if magic == MAGIC_PUB:
            if private_key is None:
                raise ValueError("this backup is encrypted to a public key: pass --key-file")
            f.seek(0)
            with tempfile.TemporaryFile() as plain:
                decrypt_stream(f, private_key, plain)
                plain.seek(0)
                with tarfile.open(fileobj=plain, mode="r:gz") as tar:
                    yield tar
            return
    if passphrase is None:
        raise ValueError("this backup is encrypted with a passphrase")
    with open_backup(path.read_bytes(), passphrase) as tar:
        yield tar


def restore_tar(tar: tarfile.TarFile, dest: Path) -> list[str]:
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


def restore(blob: bytes, passphrase: str, dest: Path) -> list[str]:
    with open_backup(blob, passphrase) as tar:
        return restore_tar(tar, dest)


# ---- the daily timer ----

def auto(src: Path, out_dir: Path, pubkey_b64: str | None, keep: int = 7,
         exclude: list[str] | None = None, now: float | None = None) -> Path | None:
    """One public-key backup into out_dir (aichain-YYYYMMDD-HHMMSS.enc, 0600), then keep only the
    newest `keep`. None (and a log line) when no public key is configured."""
    if keep < 1:
        raise ValueError("keep at least one backup")
    if not pubkey_b64:
        print("AICHAIN_BACKUP_PUBKEY is not set; skipping the backup (see docs/notes/backups.md)")
        return None
    pubkey = _decode_key(pubkey_b64, "public")
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for stale in out_dir.glob("aichain-*.enc.partial"):  # an earlier run that died
        stale.unlink()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
    final = out_dir / f"aichain-{stamp}.enc"
    partial = final.with_name(final.name + ".partial")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            manifest = create_to_pubkey(src, pubkey, f, AUTO_EXCLUDE if exclude is None else exclude, out_dir)
            f.flush()
            os.fsync(f.fileno())
        partial.replace(final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    backups = sorted(out_dir.glob("aichain-*.enc"))  # names sort by time
    for old in backups[:-keep]:
        old.unlink()
    print(f"wrote {final} ({final.stat().st_size / 1e6:.1f} MB, {len(manifest['files'])} files); "
          f"keeping {min(len(backups), keep)} backups in {out_dir}")
    return final


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


def _open_args(args: argparse.Namespace):
    with args.file.open("rb") as f:
        public = f.read(len(MAGIC_PUB)) == MAGIC_PUB
    if public:
        if args.key_file is None:
            raise SystemExit("this backup is encrypted to a public key: pass --key-file <private key file>")
        return open_backup_file(args.file, private_key=read_key_file(args.key_file))
    return open_backup_file(args.file, passphrase=_passphrase(confirm=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Encrypted backup / restore of an AIChain host")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--src", type=Path, default=ROOT)
    c.add_argument("--to-pubkey", default=None, help="encrypt to this base64 X25519 public key (no passphrase)")
    c.add_argument("--exclude", action="append", default=[], help="relative path or glob to leave out (repeat)")
    k = sub.add_parser("keygen", help="make a key pair; run it on your own computer, not the server")
    k.add_argument("--out", type=Path, required=True, help="private key file to write (0600, never overwritten)")
    a = sub.add_parser("auto", help="daily timer: back up to AICHAIN_BACKUP_PUBKEY, keep the newest N")
    a.add_argument("--dir", type=Path, default=ROOT / "backups")
    a.add_argument("--src", type=Path, default=ROOT)
    a.add_argument("--keep", type=int, default=7)
    a.add_argument("--env-file", type=Path, default=ROOT / ".env")
    a.add_argument("--include-journal", action="store_true", help="also back up the journal and traces")
    r = sub.add_parser("restore")
    r.add_argument("file", type=Path)
    r.add_argument("--to", type=Path, default=ROOT)
    r.add_argument("--key-file", type=Path, default=None, help="private key file (public-key backups)")
    v = sub.add_parser("verify")
    v.add_argument("file", type=Path)
    v.add_argument("--key-file", type=Path, default=None, help="private key file (public-key backups)")
    args = ap.parse_args(argv)
    if args.cmd == "keygen":
        public, private = keygen()
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"# AIChain backup private key (X25519). Keep it off the server; without it the\n"
                    f"# backups cannot be read. Public key: {public}\n{private}\n")
        print(f"private key written to {args.out} (keep it safe, off the server)")
        print(f"public key (put it in the server's /opt/aichain/.env):\nAICHAIN_BACKUP_PUBKEY={public}")
    elif args.cmd == "auto":
        from aichain.ops import read_env

        key = os.environ.get("AICHAIN_BACKUP_PUBKEY") or read_env(args.env_file).get("AICHAIN_BACKUP_PUBKEY")
        auto(args.src, args.dir, key, args.keep, exclude=[] if args.include_journal else None)
    elif args.cmd == "create":
        pubkey = _decode_key(args.to_pubkey, "public") if args.to_pubkey else None
        blob = None if pubkey else create(args.src, _passphrase(confirm=True), args.exclude)
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            if pubkey:
                create_to_pubkey(args.src, pubkey, f, args.exclude, args.out.resolve().parent)
            else:
                f.write(blob)
        print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    elif args.cmd == "verify":
        with _open_args(args) as tar:
            manifest = json.load(tar.extractfile("BACKUP_MANIFEST.json"))
            names = set(tar.getnames())
        missing = [n for n in manifest["files"] if n not in names]
        if missing:
            raise SystemExit(f"backup is missing {len(missing)} files, e.g. {missing[:3]}")
        print(f"ok: {len(manifest['files'])} files, created {time.ctime(manifest['created'])}")
    else:
        with _open_args(args) as tar:
            names = restore_tar(tar, args.to)
        print(f"restored {len(names)} files into {args.to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
