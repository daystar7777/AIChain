# 자동 백업 (공개키 암호화)

작성: 2026-10-13 · 코드: `aichain/backup.py` · 유닛: `deploy/systemd/aichain-backup.{service,timer}` · 테스트: `tests/test_backup.py`

## 왜 공개키인가
매일 자동으로 백업하려면 서버가 혼자 암호화할 수 있어야 한다. passphrase 방식은 passphrase를 서버에 둬야 하고, 그러면 서버를 가진 사람이 백업도 열 수 있다. 공개키 방식에서는 **서버에 공개키만 둔다.** 백업은 운영자의 X25519 공개키로 봉인되고, 여는 비밀키는 운영자 컴퓨터에만 있다. 서버가 털려도 지난 백업은 읽을 수 없다.

- 형식: 무작위 키 하나를 운영자 공개키로 봉인(SealedBox)하고, 압축한 아카이브를 그 키로 64KiB씩 XChaCha20-Poly1305(secretstream)로 암호화한다. 메모리를 거의 쓰지 않고, 잘리거나 바뀐 파일은 복원할 때 거부된다.
- SQLite DB(체인, 지식, 도구 등)는 SQLite 온라인 백업으로 복사하므로 서비스를 멈추지 않아도 일관된 사본이 들어간다.
- 매일 백업에는 저널(`data/journal`)과 trace 파일이 빠진다. 크고 관찰용일 뿐이라서다. 키·체인·기억·지식·도구·승인·설정·`.env`는 모두 들어간다. 저널까지 넣으려면 `--include-journal`.
- 공개키는 누구나 쓸 수 있으므로 공개키 백업에는 "누가 만들었는지" 증명이 없다. 백업 파일은 운영자만 쓸 수 있는 곳에 보관한다.

## 운영자가 할 일

1. **내 컴퓨터에서** 키 쌍을 만든다(서버에서 하지 않는다).
   ```bash
   python -m aichain.backup keygen --out ~/aichain-backup.key
   ```
   비밀키 파일(0600)이 생기고 화면에 `AICHAIN_BACKUP_PUBKEY=...` 한 줄이 나온다. 비밀키 파일은 비밀번호 관리자나 오프라인 저장소에 하나 더 보관한다. 잃으면 백업을 열 수 없다.
2. **서버에** 공개키를 넣고 타이머를 켠다.
   ```bash
   echo 'AICHAIN_BACKUP_PUBKEY=<위의 공개키>' >> /opt/aichain/.env
   install -d -m 700 /opt/aichain/backups
   install -m 644 deploy/systemd/aichain-backup.service deploy/systemd/aichain-backup.timer /etc/systemd/system/
   systemctl daemon-reload && systemctl enable --now aichain-backup.timer
   systemctl start aichain-backup.service && journalctl -u aichain-backup -n 5   # 첫 백업 확인
   ```
   매일 03:30 UTC쯤 `/opt/aichain/backups/aichain-YYYYMMDD-HHMMSS.enc`(0600)를 만들고 최신 7개만 남긴다. 공개키가 없으면 "AICHAIN_BACKUP_PUBKEY is not set; skipping" 한 줄만 남기고 성공으로 끝난다. 실패하면 `aichain-alert@`가 텔레그램으로 알린다.
3. **서버 밖으로 옮긴다.** 같은 디스크에 있는 백업은 디스크 고장에 소용없다. 예: 내 컴퓨터에서 매일
   ```bash
   rsync -a aichain:/opt/aichain/backups/ ~/aichain-backups/
   ```
4. **가끔 열어 본다.**
   ```bash
   python -m aichain.backup verify ~/aichain-backups/aichain-20261013-033012.enc --key-file ~/aichain-backup.key
   ```

## 복원
새 호스트에 코드를 설치한 뒤(노드 이전은 [node-migration.md](node-migration.md) 순서대로, 옛 호스트를 먼저 멈춘다):
```bash
python -m aichain.backup restore aichain-20261013-033012.enc --to /opt/aichain --key-file ~/aichain-backup.key
systemctl start aichain-bridge
```
비밀키 파일은 복원할 때만 잠깐 서버에 올리고 끝나면 지운다(또는 내 컴퓨터에서 풀어서 `data/`와 `.env`를 복사한다).

## 수동 백업
- 공개키로: `python -m aichain.backup create --to-pubkey <공개키> --out x.enc [--exclude data/journal]`
- passphrase로(기존 방식, 서버 이전용): `python -m aichain.backup create --out x.enc` — 파일 형식으로 구분하므로 `verify`/`restore`는 두 방식 모두 같은 명령이다(공개키 백업은 `--key-file`이 필요).
