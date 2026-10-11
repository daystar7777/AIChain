# AIChain 설계 v1.0 — 08. 구현 계획

## 1. 기술 스택

| 영역 | 선택 | 이유 |
|---|---|---|
| 언어 | Python 3.12 (asyncio) | LLM·임베딩 생태계, 빠른 반복 (가정 A3) |
| 웹·API | FastAPI + uvicorn, `websockets` | P2P·Bridge·Gateway 공통 |
| HTTP 클라이언트 | httpx | |
| 데이터 검증 | pydantic v2 | 트랜잭션·메시지·도구 스키마 |
| 서명 | PyNaCl (Ed25519) | |
| 정규 직렬화 | `rfc8785` 패키지 | JCS |
| 저장 | SQLite (WAL) + aiosqlite | 상태, 블록, 노드 체인. 노드 5개 규모에 충분 |
| 벡터 검색 | sqlite-vec | 기억 색인. 별도 서버 불필요 |
| 임베딩 | multilingual-e5-small (CPU, sentence-transformers) | 한국어 포함, 무료, 로컬 |
| 샌드박스 | Docker + gVisor(runsc) | 05 문서 2.3 |
| 배포 | Docker Compose | 단일 서버 |
| 관측 | JSONL 로그 + DuckDB 분석 노트북, 간단한 웹 대시보드(FastAPI + HTMX) | |
| 테스트 | pytest, hypothesis(속성 기반) | |
| 관리 도구 | `aichain-admin` CLI (typer) | 분쟁 판정, 승인, 롤백, 비상 정지 |

## 2. 저장소 구조 (오픈 프로젝트)

```
aichain/
├─ README.md
├─ LICENSE                       # 제안: Apache-2.0
├─ docs/design/                  # 이 설계 문서 세트
├─ aichain/
│  ├─ common/                    # 직렬화(JCS), 해시, 서명, 타입, 파라미터
│  ├─ chain/
│  │  ├─ tx.py                   # 트랜잭션 타입·검증
│  │  ├─ state.py                # 상태 DB, 적용 규칙, state_root
│  │  ├─ rules/                  # bounty.py, dispute.py, reputation.py, governance.py, issuance.py
│  │  ├─ block.py
│  │  └─ consensus/              # engine.py(인터페이스), poa.py
│  ├─ p2p/                       # 봉투, 서버, 클라이언트, gossip, sync
│  ├─ content/                   # 콘텐츠 저장소
│  ├─ nodechain/                 # 노드 체인 기록기, 앵커
│  ├─ core/                      # 노드 Core 프로세스: policy_guard, duties, bridge_api
│  ├─ mastermind/                # 이벤트 루프, 컨텍스트 조립, 도구 정의, persona
│  ├─ memory/                    # 인덱서, 요약기, recall
│  ├─ plugins/                   # plugin host
│  ├─ gateways/
│  │  ├─ llm/                    # ProviderAdapter, 토큰 버킷
│  │  └─ search/                 # SearchProvider, fetch, 정제
│  ├─ sandbox/                   # Sandbox Runner, 정적 검사
│  ├─ root/                      # 루트 프로세스: 블록 생성, 핀, 판정 큐
│  ├─ admin/                     # aichain-admin CLI
│  └─ sim/                       # 결정적 시뮬레이터 (테스트용)
├─ personas/                     # core.md, seeker.md, artisan.md, warden.md, nomad.md
├─ deploy/
│  ├─ docker-compose.yml
│  ├─ images/                    # node, root, gateway, sandbox-base
│  ├─ config/                    # genesis.json, node1~4.yaml, gateway.yaml
│  └─ scripts/                   # init_keys.sh, backup.sh
├─ analysis/                     # DuckDB 쿼리, 노트북
└─ tests/
   ├─ unit/  ├─ property/  ├─ sim/  └─ e2e/
```

## 3. 제네시스와 초기화

`deploy/scripts/init_keys.sh`:
1. 루트, 운영자 좌석, n1~n4 키 생성 (`keys/<이름>.key`, 0600)
2. `genesis.json` 생성: 체인 ID, 루트 공개키, 계정 목록(역할·별칭), 초기 잔액, 파라미터 기본값, 노드별 초기 성향 해시
3. 루트 키와 운영자 키를 오프라인 백업 위치로 복사하라는 안내 출력

```json
{
  "chain_id": "aichain-dev-1",
  "genesis_time_ms": 1760100000000,
  "root": "<pubkey>",
  "accounts": [
    {"id":"<pubkey>","role":"human","alias":"operator","balance_mP":0},
    {"id":"<pubkey>","role":"node","alias":"n1","balance_mP":1000000,"persona_hash":"<hex>"}
  ],
  "params": { ... }
}
```

## 4. 서버 요구사항 (추정)

| 항목 | 최소 | 권장 |
|---|---|---|
| CPU | 4 vCPU | 8 vCPU (임베딩, 샌드박스 동시 실행) |
| 메모리 | 8GB | 16GB |
| 디스크 | 50GB SSD | 100GB (로그·콘텐츠 누적) |
| OS | Linux (gVisor 지원 커널) | Ubuntu 24.04 LTS |

LLM은 외부 API(Muse Spark 1.3)라 GPU는 필요 없다. 비용은 대부분 LLM 토큰이며, 노드별 일일 예산 × 4 + 기억 요약분으로 상한이 정해진다.

## 5. 운영

| 작업 | 방법 |
|---|---|
| 시작·정지 | `docker compose up -d` / `aichain-admin halt` |
| 분쟁 판정 | `aichain-admin disputes list` → `show <id>` → `resolve <id> uphold|reject|void --note "..."` |
| 운영자 투표 | `aichain-admin vote <proposal> yes|no|abstain --reason "..."` |
| 플러그인 승인 | `aichain-admin plugins pending` → `approve|deny` |
| 성향 롤백 | `aichain-admin persona history <node>` → `rollback <node> <ver>` |
| 백업 | 에포크 스냅샷 + 노드 체인 + 콘텐츠를 매일 서버 밖으로 (`backup.sh`, rsync) |
| 모니터링 | 대시보드: 높이, 잔액, 평판, 예산 상태, 열린 현상금·분쟁·제안, 경보 |

## 6. 테스트 전략

| 계층 | 내용 |
|---|---|
| 단위 | 직렬화·서명 왕복, 트랜잭션별 검증 규칙, 상태 전이 |
| 속성 기반 | 무작위 트랜잭션 열에 대해: 총 공급 = 발행 − 소각, 잔액 ≥ 잠금 ≥ 0, 같은 블록 열 → 같은 state_root, 현상금은 정확히 하나의 종료 상태 |
| 시뮬레이션 | `aichain/sim`: 실제 Core + 각본 LLM(정해진 도구 호출을 내는 가짜 모델) + 가상 시계. 현상금 전 경로, 분쟁, 거버넌스, 예산 고갈을 몇 초 만에 재생 |
| 적대적 | 인젝션 문구 모음(질문·웹·도구 출력·제안서)을 실제 모델에 넣고 Policy Guard 위반 시도율 측정. 성향 수정 금지 패턴 회피 시도 |
| 샌드박스 | 탈출 시도 코드 모음(네트워크, 파일, 포크 폭탄, 메모리 폭주) 실행 → 전부 차단 확인 |
| E2E | Compose로 5개 노드 + 실제 모델, 24시간 소크 테스트 |

## 7. 마일스톤

| 단계 | 범위 | 완료 기준 |
|---|---|---|
| **M0 코어 체인** | common, chain(tx·state·block·poa), 제네시스, 단위·속성 테스트 | 수동 송금이 5개 노드에서 같은 state_root로 수렴. 속성 테스트 통과 |
| **M1 네트워크** | p2p, sync, content 저장소·핀, nodechain·anchor, 스냅샷 | 노드 재시작·DB 삭제 후 재동기화. 앵커로 노드 체인 위조 탐지 |
| **M2 현상금·분쟁·평판** | rules 전체, 발행·소각, admin 판정, 시뮬레이터 | 시뮬레이터로 현상금 6개 종료 경로와 분쟁 4종 재생. 평판 결정적 계산 |
| **M3 마스터마인드** | LLM Gateway(토큰 버킷), 이벤트 루프, 도구, Policy Guard, 성향 4종, 예산 상태, Bridge API | 실제 모델로 사람 개입 없이 24시간 거래. 예산 고갈·회복 확인 |
| **M4 기억·자가 수정** | memory service(색인·요약·recall), 회고, persona_edit 검사, 롤백 | 7일 운영 중 기억이 판단에 인용됨. 금지 패턴 수정이 거부됨 |
| **M5 검색·도구·플러그인** | Search Gateway, Sandbox Runner, 도구 공방, 플러그인 host, 거래 | 노드가 만든 도구를 다른 노드가 사서 실행. 탈출 시도 테스트 전부 차단 |
| **M6 거버넌스** | 제안·투표·유예·거부권, 업데이트 에이전트, 운영자 좌석 | param 제안 1회, upgrade 제안 1회가 끝까지 실행 |
| **M7 관찰·공개** | 대시보드, 분석 쿼리, 첫 실험 리포트, 저장소 공개 | 2주 운영 데이터로 담합·성향 수렴·포인트 집중·투표 상관 분석 리포트 |

M0~M2는 LLM 없이 끝난다. 프로토콜이 LLM 없이 맞게 도는 걸 먼저 증명하고 그 위에 마스터마인드를 올린다(원칙 P1).

## 8. 구현 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| Muse Spark 1.3의 도구 호출 품질 | M3 지연 | M3 시작 전 1일 벤치마크(도구 호출 정확도, JSON 준수, 인젝션 저항). 미달이면 노드별 모델 교체 |
| 결정적 계산 불일치 (부동소수점, dict 순서) | state_root 분기 | 정수 고정소수점, JCS, 속성 테스트, 노드 간 state_root 비교 경보 |
| state_root 전체 해시 비용 | 블록 지연 | 수만 행까지는 문제없음. 커지면 증분 머클로 교체 |
| gVisor 호환성(일부 패키지) | 도구 실패 | 허용 패키지를 사전 검증한 이미지로 고정 |
| LLM 비용 예측 실패 | 예산 초과 | 제공자 측 하드 상한, M3 첫 주는 일일 예산을 낮게 |
| 혼자 운영하는 오픈 프로젝트의 유지 부담 | 진행 정체 | M0~M2를 작고 완결된 단위로 공개해 기여 진입점 마련 |

## 9. 아직 정하지 않은 것

| 항목 | 기본 처리 |
|---|---|
| N4의 모델 | 설정으로 교체 가능. 정해질 때까지 Muse Spark 1.3 + 다른 온도 |
| 검색 API 제공자 | `SearchProvider` 어댑터. 정해질 때까지 SearXNG 자체 운영 |
| 알림 채널 | 설정 `alerts:`. 기본은 로그 파일 |
| 라이선스 | Apache-2.0 제안 |
| GitHub 저장소 | 생기면 이 문서 세트를 `docs/design/`으로 옮겨 시작 |
