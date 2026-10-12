"""Persistent, signed chain (M0, design docs 01–02) in SQLite.

- Every state-changing board call becomes a transaction envelope
  {v, chain_id, type, from, nonce, valid_until, body}, tx_id = sha256(JCS(envelope)),
  signed by the sender's key (Core holds keys; the Mastermind never does).
- The root produces a block at every height: {v, chain_id, height, prev_hash, timestamp_ms,
  epoch, tx_root, state_root, proposer}, block_hash = sha256(JCS(header)), signed by the root.
  A block includes the transactions made since the previous block.
- On start the log is replayed: every signature, nonce, block link and tx_root is checked,
  and state roots are compared (every block when written; on replay every
  ROOT_CHECK_EVERY blocks and at the tip, to keep startup fast).

Content bodies and each node's sealed (not yet revealed) answers are stored too.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from aichain.board import Board, BoardError
from aichain.chain.crypto import KeyStore, hash_obj, merkle_root, verify
from aichain.chain.errors import ChainHalted

CHAIN_ID = "aichain-dev-1"
ROOT_ACCOUNT = "root"
ROOT_VERSION = "v12"  # bump when the state-root definition changes
ROOT_CHECK_EVERY = 120
CHECKPOINT_EVERY = ROOT_CHECK_EVERY  # kept for callers/tests
VALID_FOR_BLOCKS = 120_960

# Which body field names the signer of each transaction type.
SENDER_FIELD = {"open_account": None, "create_bounty": "asker", "commit_answer": "author",
                "reveal_answer": "author", "reject_answer": "asker", "accept_answer": "asker",
                "heartbeat": "account", "publish_tool": "author", "withdraw_tool": "author",
                "license_tool": "buyer", "pay_tool_use": "user", "open_dispute": "opener",
                "resolve_dispute": None, "rate": "rater", "create_proposal": "proposer",
                "vote": "voter", "veto_proposal": None, "create_tool_request": "buyer",
                "submit_tool": "seller", "verify_submission": None, "create_offer": "seller",
                "withdraw_offer": "seller", "buy_offer": "buyer", "deliver_offer": "seller",
                "confirm_delivery": "buyer", "subscribe": "subscriber", "sub_deliver": "seller",
                "cancel_subscription": "subscriber", "create_loan_request": "borrower",
                "cancel_loan_request": "borrower", "fund_loan": "lender", "repay_loan": "borrower"}


class ReplayMismatch(Exception):
    """The log does not verify (bad signature, nonce, block link, tx root or state root)."""


def sender_of(tx_type: str, body: dict[str, Any]) -> str:
    field = SENDER_FIELD.get(tx_type, "missing")
    if field == "missing":
        raise ReplayMismatch(f"unknown transaction type {tx_type}")
    return ROOT_ACCOUNT if field is None else body[field]


class ChainStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS log (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, height INTEGER NOT NULL,
        id TEXT NOT NULL, data TEXT NOT NULL, sig TEXT NOT NULL, ts REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS content (hash TEXT PRIMARY KEY, author TEXT, body TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS private (
        node TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY (node, key));
    CREATE TABLE IF NOT EXISTS sealed (
        answer_id TEXT PRIMARY KEY, node TEXT NOT NULL, body TEXT NOT NULL, salt TEXT NOT NULL);
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)
        self._lock = threading.Lock()

    def _write(self, sql: str, args: tuple) -> None:
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._write("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    # ---- log ----

    def append(self, kind: str, height: int, entry_id: str, data: dict[str, Any], sig: str) -> None:
        self._write("INSERT INTO log (kind, height, id, data, sig, ts) VALUES (?, ?, ?, ?, ?, ?)",
                    (kind, height, entry_id, json.dumps(data, ensure_ascii=False), sig, time.time()))

    def entries(self) -> list[tuple[str, int, str, dict[str, Any], str]]:
        with self._lock:
            rows = self._conn.execute("SELECT kind, height, id, data, sig FROM log ORDER BY seq").fetchall()
        return [(k, h, i, json.loads(d), s) for k, h, i, d, s in rows]

    def tx_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM log WHERE kind = 'tx'").fetchone()[0]

    # ---- content and sealed answers ----

    def put_content(self, h: str, obj: Any, author: str | None) -> None:
        self._write("INSERT OR IGNORE INTO content VALUES (?, ?, ?)",
                    (h, author, json.dumps(obj, ensure_ascii=False)))

    def all_content(self) -> list[tuple[str, str | None, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT hash, author, body FROM content").fetchall()
        return [(h, a, json.loads(b)) for h, a, b in rows]

    def put_sealed(self, node: str, answer_id: str, body: dict[str, Any], salt: str) -> None:
        self._write("INSERT OR REPLACE INTO sealed VALUES (?, ?, ?, ?)",
                    (answer_id, node, json.dumps(body, ensure_ascii=False), salt))

    def delete_sealed(self, answer_id: str) -> None:
        self._write("DELETE FROM sealed WHERE answer_id = ?", (answer_id,))

    def sealed_for(self, node: str) -> dict[str, tuple[dict[str, Any], str]]:
        with self._lock:
            rows = self._conn.execute("SELECT answer_id, body, salt FROM sealed WHERE node = ?",
                                      (node,)).fetchall()
        return {a: (json.loads(b), s) for a, b, s in rows}

    def put_private(self, node: str, key: str, body: Any) -> None:
        """Node-private data (e.g. the plaintext behind an info offer); never on the chain."""
        self._write("INSERT OR REPLACE INTO private VALUES (?, ?, ?)",
                    (node, key, json.dumps(body, ensure_ascii=False)))

    def private_for(self, node: str) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute("SELECT key, body FROM private WHERE node = ?", (node,)).fetchall()
        return {k: json.loads(b) for k, b in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class ChainState:
    """Verification state carried through replay and live operation."""

    def __init__(self, root_pubkey: str) -> None:
        self.root_pubkey = root_pubkey
        self.nonces: dict[str, int] = {}
        self.prev_hash = "0" * 64
        self.pending: list[str] = []  # tx ids made since the last block

    def pubkey_of(self, board: Board, account: str) -> str | None:
        if account == ROOT_ACCOUNT:
            return self.root_pubkey
        acct = board.accounts.get(account)
        return acct.pubkey if acct else None


def _check_tx(board: Board, st: ChainState, env: dict[str, Any], tx_id: str, sig: str) -> None:
    if hash_obj(env) != tx_id:
        raise ReplayMismatch(f"tx {tx_id[:12]}: id does not match its envelope")
    if env.get("chain_id") != CHAIN_ID or env.get("v") != 1:
        raise ReplayMismatch(f"tx {tx_id[:12]}: wrong chain or version")
    sender = env["from"]
    if sender_of(env["type"], env["body"]) != sender:
        raise ReplayMismatch(f"tx {tx_id[:12]}: signer is not the party the transaction acts for")
    pub = st.pubkey_of(board, sender)
    if not pub or not verify(pub, "tx", tx_id, sig):
        raise ReplayMismatch(f"tx {tx_id[:12]}: bad signature from {sender}")
    if env["nonce"] != st.nonces.get(sender, 0) + 1:
        raise ReplayMismatch(f"tx {tx_id[:12]}: nonce {env['nonce']} out of order for {sender}")
    if env["valid_until"] < board.height:
        raise ReplayMismatch(f"tx {tx_id[:12]}: expired")


def _block_header(board: Board, st: ChainState, now_ms: int) -> dict[str, Any]:
    return {"v": 1, "chain_id": CHAIN_ID, "height": board.height, "prev_hash": st.prev_hash,
            "timestamp_ms": now_ms, "epoch": board.height // 720,
            "tx_root": merkle_root(st.pending), "state_root": f"{ROOT_VERSION}:{board.state_root()}",
            "proposer": ROOT_ACCOUNT}


def replay(board: Board, store: ChainStore, st: ChainState | None = None) -> dict[str, int]:
    st = st or ChainState(store.get_meta("root_pubkey") or "")
    txs = blocks = roots = 0
    entries = store.entries()
    last_block = max((i for i, e in enumerate(entries) if e[0] == "block"), default=-1)
    for idx, (kind, height, entry_id, data, sig) in enumerate(entries):
        if kind == "tx":
            _check_tx(board, st, data, entry_id, sig)
            try:
                board.apply(data["type"], data["body"])
            except BoardError as e:
                raise ReplayMismatch(f"tx {entry_id[:12]} no longer applies: {e}") from e
            st.nonces[data["from"]] = data["nonce"]
            st.pending.append(entry_id)
            txs += 1
        elif kind == "block":
            if hash_obj(data) != entry_id or not verify(st.root_pubkey, "block", entry_id, sig):
                raise ReplayMismatch(f"block {height}: bad hash or root signature")
            if data["prev_hash"] != st.prev_hash or data["height"] != board.height + 1:
                raise ReplayMismatch(f"block {height}: does not extend the chain")
            if data["tx_root"] != merkle_root(st.pending):
                raise ReplayMismatch(f"block {height}: transactions do not match tx_root")
            board.advance(1)
            version, _, digest = data["state_root"].rpartition(":")
            if (version == ROOT_VERSION and (height % ROOT_CHECK_EVERY == 0 or idx == last_block)
                    and board.state_root() != digest):
                raise ReplayMismatch(f"block {height}: state root differs")
            roots += version == ROOT_VERSION and (height % ROOT_CHECK_EVERY == 0 or idx == last_block)
            st.prev_hash, st.pending = entry_id, []
            blocks += 1
    return {"transactions": txs, "blocks": blocks, "state_roots_checked": int(roots),
            "height": board.height}


def attach(board: Board, store: ChainStore, keys: KeyStore | None = None) -> dict[str, int]:
    """Replay, then sign and record every new transaction and produce a signed block per height."""
    if keys is None:
        keys = KeyStore(store.path.parent / "keys")
    root_pub = store.get_meta("root_pubkey")
    if root_pub is None:
        root_pub = keys.pubkey(ROOT_ACCOUNT)
        store.set_meta("root_pubkey", root_pub)
        store.set_meta("chain_id", CHAIN_ID)
    elif root_pub != keys.pubkey(ROOT_ACCOUNT):
        raise ReplayMismatch("the root key does not match the one this chain was created with")
    st = ChainState(root_pub)
    stats = replay(board, store, st)
    board.chain_state = st

    def record(tx_type: str, body: dict[str, Any]) -> None:
        try:
            if tx_type == "open_account" and not body.get("pubkey"):
                # Bind a signing key to every account so its later transactions can be verified.
                body = {**body, "pubkey": keys.pubkey(body["account_id"])}
                board.accounts[body["account_id"]].pubkey = body["pubkey"]
            sender = sender_of(tx_type, body)
            env = {"v": 1, "chain_id": CHAIN_ID, "type": tx_type, "from": sender,
                   "nonce": st.nonces.get(sender, 0) + 1,
                   "valid_until": board.height + VALID_FOR_BLOCKS, "body": body}
            tx_id = hash_obj(env)
            store.append("tx", board.height, tx_id, env, keys.sign(sender, "tx", tx_id))
            st.nonces[sender] = env["nonce"]
            st.pending.append(tx_id)
        except Exception as e:  # disk full, locked or corrupt DB, missing key: fail closed
            board.halted = f"chain log write failed: {type(e).__name__}: {e}"
            raise ChainHalted(board.halted) from e

    board.recorder = record
    original = board._on_block

    def on_block():
        events = original()
        try:
            header = _block_header(board, st, int(time.time() * 1000))
            block_hash = hash_obj(header)
            store.append("block", board.height, block_hash, header,
                         keys.sign(ROOT_ACCOUNT, "block", block_hash))
            st.prev_hash, st.pending = block_hash, []
        except Exception as e:
            board.halted = f"chain log write failed: {type(e).__name__}: {e}"
            raise ChainHalted(board.halted) from e
        return events

    board._on_block = on_block
    return stats
