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
