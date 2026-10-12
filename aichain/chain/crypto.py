"""Keys and signatures (design doc 01, ch. 1–2): Ed25519 via PyNaCl, domain-separated.

Signed bytes are b"aichain/<kind>/v1\\n" + the 32-byte hash, so a signature made for a
transaction can never be replayed as a block signature or the other way round.
Private keys live only in the key directory (0700, files 0600) and are used by Core;
the Mastermind never sees them.
"""

from __future__ import annotations

import os
from pathlib import Path

import base64

from nacl.exceptions import BadSignatureError, CryptoError
from nacl.public import SealedBox
from nacl.signing import SigningKey, VerifyKey

from aichain.common import canonical_json, sha256_hex


def signed_bytes(kind: str, digest_hex: str) -> bytes:
    return f"aichain/{kind}/v1\n".encode() + bytes.fromhex(digest_hex)


def verify(pubkey_hex: str, kind: str, digest_hex: str, sig_hex: str) -> bool:
    try:
        VerifyKey(bytes.fromhex(pubkey_hex)).verify(signed_bytes(kind, digest_hex), bytes.fromhex(sig_hex))
        return True
    except (BadSignatureError, ValueError):
        return False


def hash_obj(obj: object) -> str:
    return sha256_hex(canonical_json(obj))


def merkle_root(ids: list[str]) -> str:
    """Binary Merkle root of hex ids (01 doc 4: duplicate the last on odd levels)."""
    if not ids:
        return sha256_hex(b"")
    level = [bytes.fromhex(i) for i in ids]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [bytes.fromhex(sha256_hex(level[i] + level[i + 1])) for i in range(0, len(level), 2)]
    return level[0].hex()


def seal_for(pubkey_hex: str, plaintext: bytes) -> str:
    """End-to-end encryption to an account (X25519 derived from its Ed25519 key). Returns base64."""
    box = SealedBox(VerifyKey(bytes.fromhex(pubkey_hex)).to_curve25519_public_key())
    return base64.b64encode(box.encrypt(plaintext)).decode()


class DecryptError(Exception):
    pass


class KeyStore:
    def __init__(self, directory: Path | None) -> None:
        """directory None: keys live only in memory (tests, networks without a chain)."""
        self.dir = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o700)
        self._keys: dict[str, SigningKey] = {}

    def open_sealed(self, name: str, ciphertext_b64: str) -> bytes:
        try:
            box = SealedBox(self.key(name).to_curve25519_private_key())
            return box.decrypt(base64.b64decode(ciphertext_b64))
        except (CryptoError, ValueError) as e:
            raise DecryptError(f"cannot decrypt for {name}") from e

    def key(self, name: str) -> SigningKey:
        if name not in self._keys and self.dir is None:
            self._keys[name] = SigningKey.generate()
        if name not in self._keys:
            path = self.dir / f"{name}.key"
            if path.exists():
                self._keys[name] = SigningKey(bytes.fromhex(path.read_text().strip()))
            else:
                k = SigningKey.generate()
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(k.encode().hex())
                self._keys[name] = k
        return self._keys[name]

    def pubkey(self, name: str) -> str:
        return self.key(name).verify_key.encode().hex()

    def sign(self, name: str, kind: str, digest_hex: str) -> str:
        return self.key(name).sign(signed_bytes(kind, digest_hex)).signature.hex()
