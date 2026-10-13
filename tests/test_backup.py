import pytest
from nacl.exceptions import CryptoError

from aichain.backup import create, open_backup, restore
from aichain.chain.store import ChainStore
from aichain.memory import KnowledgeStore
from aichain.sim.factory import build_network


def test_backup_restore_moves_a_whole_node_host(tmp_path):
    old = tmp_path / "old"
    data = old / "data"
    net = build_network("scripted", chain=ChainStore(data / "chain" / "chain.db"),
                        knowledge=KnowledgeStore(data / "knowledge" / "knowledge.db"))
    net.nodes["n1"].memory.remember(0, "fact", "self", 1.0, "기억해 둘 사실")
    net.advance(5)
    root_pub = net.keys.pubkey("root")
    (old / ".env").write_text("OPENCODE_GO_API_KEY=x\n")
    blob = create(old, "correct horse battery")
    assert b"OPENCODE_GO_API_KEY" not in blob and b"root" not in blob[:200]
    with pytest.raises(CryptoError):
        open_backup(blob, "wrong passphrase!!")

    new = tmp_path / "new"
    new.mkdir()
    names = restore(blob, "correct horse battery", new)
    assert "data/chain/keys/root.key" in names
    assert oct((new / "data/chain/keys/root.key").stat().st_mode)[-3:] == "600"
    moved = build_network("scripted", chain=ChainStore(new / "data" / "chain" / "chain.db"),
                          knowledge=KnowledgeStore(new / "data" / "knowledge" / "knowledge.db"))
    assert moved.keys.pubkey("root") == root_pub  # same identity on the new host
    assert moved.board.height == 5 and moved.board.state_root() == net.board.state_root()
    assert any(m.content == "기억해 둘 사실" for m in moved.nodes["n1"].memory.records)


def make_host(root):
    data = root / "data"
    net = build_network("scripted", chain=ChainStore(data / "chain" / "chain.db"),
                        knowledge=KnowledgeStore(data / "knowledge" / "knowledge.db"))
    net.nodes["n1"].memory.remember(0, "fact", "self", 1.0, "기억해 둘 사실")
    net.advance(5)
    (root / ".env").write_text("OPENCODE_GO_API_KEY=x\n")
    (data / "journal").mkdir()
    (data / "journal" / "journal.db").write_bytes(b"journal")
    (data / "bridge-20261012.jsonl").write_text("{}\n")
    return net


def test_public_key_backup_needs_only_the_key_file_to_restore(tmp_path, capsys):
    from aichain.backup import main

    key = tmp_path / "op" / "backup.key"
    key.parent.mkdir()
    assert main(["keygen", "--out", str(key)]) == 0
    pub = capsys.readouterr().out.split("AICHAIN_BACKUP_PUBKEY=")[1].strip()
    assert oct(key.stat().st_mode)[-3:] == "600" and pub in key.read_text()
    with pytest.raises(FileExistsError):  # never overwrites a private key
        main(["keygen", "--out", str(key)])

    old = tmp_path / "old"
    net = make_host(old)  # still running: its databases are open (WAL)
    out = tmp_path / "b.enc"
    assert main(["create", "--to-pubkey", pub, "--out", str(out), "--src", str(old)]) == 0
    blob = out.read_bytes()
    assert blob.startswith(b"AICHAINBK2") and b"OPENCODE_GO_API_KEY" not in blob
    assert main(["verify", str(out), "--key-file", str(key)]) == 0
    assert "ok:" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--key-file"):
        main(["verify", str(out)])
    other = tmp_path / "other.key"
    main(["keygen", "--out", str(other)])
    with pytest.raises(CryptoError):
        main(["verify", str(out), "--key-file", str(other)])

    new = tmp_path / "new"
    new.mkdir()
    assert main(["restore", str(out), "--to", str(new), "--key-file", str(key)]) == 0
    assert oct((new / "data/chain/keys/root.key").stat().st_mode)[-3:] == "600"
    assert not list(new.rglob("*-wal"))  # databases come as consistent copies
    moved = build_network("scripted", chain=ChainStore(new / "data" / "chain" / "chain.db"),
                          knowledge=KnowledgeStore(new / "data" / "knowledge" / "knowledge.db"))
    assert moved.keys.pubkey("root") == net.keys.pubkey("root")
    assert moved.board.height == 5 and moved.board.state_root() == net.board.state_root()
    assert any(m.content == "기억해 둘 사실" for m in moved.nodes["n1"].memory.records)


def test_public_key_stream_is_chunked_and_detects_tampering(tmp_path):
    import io
    import os

    from aichain.backup import CHUNK, create_to_pubkey, decrypt_stream, keygen
    from aichain.backup import _decode_key as decode

    src = tmp_path / "src"
    (src / "data").mkdir(parents=True)
    (src / "data" / "blob.bin").write_bytes(os.urandom(5 * CHUNK))  # incompressible
    pub, priv = keygen()
    buf = io.BytesIO()
    create_to_pubkey(src, decode(pub, "public"), buf)
    blob = buf.getvalue()
    plain = io.BytesIO()
    decrypt_stream(io.BytesIO(blob), decode(priv, "private"), plain)
    assert plain.getbuffer().nbytes > 5 * CHUNK
    with pytest.raises(ValueError, match="truncated"):  # a chunk boundary: the final tag is missing
        decrypt_stream(io.BytesIO(blob[:10 + 80 + 24 + 4 + CHUNK + 17]), decode(priv, "private"), io.BytesIO())
    bad = bytearray(blob)
    bad[len(blob) // 2] ^= 1
    with pytest.raises(CryptoError):
        decrypt_stream(io.BytesIO(bytes(bad)), decode(priv, "private"), io.BytesIO())


def test_live_sqlite_writes_are_in_the_backup(tmp_path):
    import io
    import sqlite3
    import tarfile

    from aichain.backup import write_archive

    src = tmp_path / "src"
    (src / "data").mkdir(parents=True)
    conn = sqlite3.connect(src / "data" / "live.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")  # everything stays in the -wal file
    conn.execute("CREATE TABLE t (x)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(100)])
    conn.commit()
    assert (src / "data" / "live.db-wal").stat().st_size > 0
    buf = io.BytesIO()
    manifest = write_archive(src, buf)
    assert manifest["files"] == ["data/live.db"]
    out = tmp_path / "out"
    with tarfile.open(fileobj=io.BytesIO(buf.getvalue()), mode="r:gz") as tar:
        tar.extractall(out, filter="data")
    assert sqlite3.connect(out / "data" / "live.db").execute("SELECT COUNT(*) FROM t").fetchone() == (100,)


def test_daily_auto_backup_keeps_the_newest_seven(tmp_path, capsys):
    from aichain.backup import auto, keygen, main, open_backup_file, read_key_file

    src = tmp_path / "host"
    make_host(src)
    dest = tmp_path / "backups"
    assert auto(src, dest, None) is None and not dest.exists()  # no key: skip, say so
    assert "AICHAIN_BACKUP_PUBKEY is not set" in capsys.readouterr().out
    pub, priv = keygen()
    dest.mkdir()
    (dest / "aichain-20260101-000000.enc.partial").write_bytes(b"died")
    made = [auto(src, dest, pub, keep=7, now=1_790_000_000 + i * 86400) for i in range(9)]
    left = sorted(p.name for p in dest.iterdir())
    assert left == sorted(p.name for p in made[2:]) and all(n.endswith(".enc") for n in left)
    assert oct(made[-1].stat().st_mode)[-3:] == "600"
    keyfile = tmp_path / "k"
    keyfile.write_text(f"# comment\n{priv}\n")
    with open_backup_file(made[-1], private_key=read_key_file(keyfile)) as tar:
        names = tar.getnames()
    assert "data/chain/keys/root.key" in names and ".env" in names
    assert not any(n.startswith("data/journal") or n.startswith("data/bridge-") for n in names)
    env = tmp_path / ".env"
    env.write_text(f"AICHAIN_BACKUP_PUBKEY={pub}\n")
    assert main(["auto", "--src", str(src), "--dir", str(dest), "--keep", "2", "--env-file", str(env)]) == 0
    assert len(list(dest.glob("*.enc"))) == 2
