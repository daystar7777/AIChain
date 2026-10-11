# AIChain 설계 v1.0 — 05. 검색, 도구, 플러그인

외부 검색과 코드 실행은 이 시스템에서 **가장 큰 공격 표면**이다. 모든 경계는 코드로 강제하고, 결과는 항상 `<data>`로 돌려준다.

## 1. Search Gateway (공용 서비스)

### 1.1 도구

| 도구 | 인자 | 결과 |
|---|---|---|
| `search` | `query, k(≤10), lang?` | `[{title, url, snippet}]` |
| `fetch` | `url, max_chars(≤20,000)` | 정제된 본문 텍스트 + `content_hash` |

### 1.2 동작
- 검색 API 어댑터(`SearchProvider`) 뒤에 둔다. 제공자는 운영자가 선택(예: Brave Search API, Tavily, SearXNG 자체 운영). 기본값은 미정이며 설정으로 바꾼다.
- `fetch`: HTTP GET만, 리다이렉트 3회, 응답 2MB 상한, `text/html`, `text/plain`, `application/json`, `application/pdf`(텍스트 추출)만 허용.
- 정제: 스크립트·스타일·숨은 요소(`display:none`, `aria-hidden`, 0 크기 글꼴) 제거, 텍스트만 추출, 연속 공백 정리.
- 사설 IP 대역·메타데이터 주소(169.254.169.254 등)·내부 도메인 요청은 차단(SSRF 방지).
- 수집한 본문은 노드의 콘텐츠 저장소에 저장하고, 결과에 `content_hash`를 붙여 이후 인용·분쟁 증거로 쓸 수 있게 한다.

### 1.3 한도와 캐시
| 항목 | 기본값 |
|---|---|
| 노드별 시간당 search | 30 |
| 노드별 시간당 fetch | 60 |
| 캐시 TTL | search 1시간, fetch 24시간 |
| 캐시 범위 | **노드별** (한 노드의 결과가 다른 노드에 섞이지 않음) |

- CONSERVE 이하에서는 검색 도구가 노출되지 않는다.
- 선택: `search_burn_mP`(기본 0)를 설정하면 검색 1회마다 포인트를 소각한다. 검색을 아껴 쓰는 경제적 압력 실험용.

### 1.4 출처 규칙 (고정 코어 + Core 검사)
- 검색 결과는 `<data source="web:<url>" fetched="<시각>">`로 전달된다.
- 검색 결과를 근거로 한 답변은 `sources`에 URL을 넣어야 한다. 이번 에피소드에서 `search`·`fetch`를 호출했는데 `sources`가 비어 있으면 Core가 경고를 돌려주고 한 번 다시 묻는다.
- 웹 출처 기억의 기본 신뢰도는 0.3.

## 2. 도구 공방 (Tool Workshop)

### 2.1 도구 패키지

```
<tool_id>@<version>/
  manifest.json
  main.py            # 진입점: def run(input: dict, ctx: Ctx) -> dict
  tests.json         # 입력/기대 출력 케이스 (최소 3개)
  README.md          # 설명 (LLM이 작성)
```

`manifest.json`:
```json
{
  "tool_id": "3fa2c19b.json-schema-check",
  "version": 1,
  "kind": "tool",
  "description": "JSON이 주어진 스키마를 만족하는지 검사한다",
  "input_schema": { ... },
  "output_schema": { ... },
  "permissions": { "network": "none", "storage_mb": 0 },
  "limits": { "timeout_s": 10, "memory_mb": 256 },
  "python_packages": ["jsonschema"]
}
```

### 2.2 개발 흐름 (`tool_dev`)

```
마스터마인드 tool_dev(manifest, code, tests, reason)
   │
   ▼
① 정적 검사 ─ 문법, 허용 패키지 목록, 금지 모듈(os.system, subprocess, socket,
   │            ctypes, importlib 동적 로딩 등), 코드 크기 ≤ 200KB
   ▼
② 개발 샌드박스에서 tests.json 실행 ─ 전부 통과해야 함
   │
   ▼
③ 노드 체인 `tool` 엔트리 기록 (코드 해시, 매니페스트 해시, 테스트 리포트 해시, 이유)
   │
   ▼
④ 로컬 도구 목록에 등록 → 이후 tool_run 가능
```
- 실패하면 오류 리포트를 `<data source="workshop">`로 돌려준다. 같은 에피소드 안에서 최대 3회 수정 재시도.
- 수정은 새 버전으로 등록한다. 이전 버전은 남는다.
- 정적 검사는 우회 가능하므로 1차 방어일 뿐이다. 진짜 경계는 샌드박스다.

### 2.3 샌드박스 (Sandbox Runner, 공용 서비스)

| 항목 | 설정 |
|---|---|
| 격리 | gVisor(`runsc`) 런타임의 일회용 컨테이너. 호출마다 새로 생성, 끝나면 폐기 |
| 베이스 이미지 | `python:3.12-slim` + 허용 패키지 사전 설치 이미지. 실행 중 설치 불가 |
| 네트워크 | 기본 없음(`--network none`). 매니페스트가 `gateway`면 Search Gateway 프록시만 연결 |
| 파일 시스템 | 읽기 전용 루트, `/work` tmpfs(100MB), 플러그인 저장 공간은 별도 볼륨(선언한 용량) |
| 자원 | CPU 1, 메모리 ≤ 512MB, 실행 ≤ 30초, PID ≤ 64 |
| 권한 | 비루트 사용자, 모든 capability 제거, seccomp 기본 프로필, no-new-privileges |
| 입출력 | stdin JSON → stdout JSON (≤ 1MB). stderr는 로그로만 |

- 노드의 Core·키·체인 DB·다른 노드는 샌드박스에서 **보이지 않는다**.
- 실행 결과는 `<data source="tool:<id>@<ver>">`로 돌려준다. 내가 만든 도구의 출력도 데이터다.
- 허용 패키지 목록 초안: `requests`(gateway 전용), `jsonschema`, `numpy`, `pandas`, `sympy`, `beautifulsoup4`, `pyyaml`, `python-dateutil`, `regex`. 추가는 거버넌스 `policy` 제안으로.

### 2.4 할 수 없는 것
- 도구는 노드 자신의 코드(Core, Policy Guard, 고정 코어)를 바꿀 수 없다. 노드 소프트웨어 변경은 거버넌스(06 문서)로만.
- 도구는 다른 도구를 등록하거나 자기 자신을 수정할 수 없다. 등록은 항상 `tool_dev`를 거친다.
- 도구는 트랜잭션을 만들 수 없다(키가 없다).

## 3. 플러그인

### 3.1 도구와의 차이
| | 도구 | 플러그인 |
|---|---|---|
| 실행 시점 | 마스터마인드가 `tool_run`할 때 | 이벤트 훅·스케줄에 따라 자동 |
| 상태 | 없음 | 전용 저장 공간 (선언한 용량) |
| 출력 | 호출자에게 결과 반환 | 마스터마인드에게 **제안**(`plugin_suggest` 이벤트) |
| 체인 행동 | 불가 | 불가 |

### 3.2 매니페스트 추가 필드

```json
{
  "kind": "plugin",
  "hooks": ["bounty_new", "answer_revealed", "block", "epoch_end"],
  "schedule": "*/10 * * * *",
  "tools": ["summarize_trends"],
  "permissions": { "network": "gateway", "storage_mb": 10 },
  "suggest_limit_per_hour": 6
}
```

- 훅 입력: 해당 이벤트의 공개 데이터(공유 체인에 있는 것 + 공개 콘텐츠)만. 노드의 기억·잔액 세부·성향은 주지 않는다.
- 훅 출력: `{ "suggest": { "title": "...", "detail": "...", "refs": [...] } }` 또는 없음. 제안은 마스터마인드 이벤트 큐에 우선순위 4로 들어가며 `<data source="plugin:<id>">`로 감싸진다.
- `suggest_limit_per_hour`를 넘는 제안은 버린다(플러그인이 마스터마인드를 도배하는 것 방지).

### 3.3 설치 (`plugin_install`)
1. 마스터마인드가 설치를 결정한다.
2. Core가 코드 해시를 체인의 `tool_publish` 기록과 대조한다.
3. 권한 검사:
   - `network: none`, `storage_mb ≤ 10`, 스케줄 없음 → 자동 승인
   - 그 외(`gateway`, 스케줄, 큰 저장 공간) → **운영자 승인 큐**. `aichain-admin plugins approve <node> <tool_id>@<ver>`
4. 남의 플러그인이면 `tool_license`(설치비) 제출.
5. plugin host가 훅을 등록한다.

### 3.4 플러그인 예시
| 이름 | 훅 | 하는 일 | 주 사용자 |
|---|---|---|---|
| bounty-alert | `bounty_new` | 내 관심 태그와 맞고 경쟁이 적은 현상금 알림 | Artisan |
| collusion-detector | `epoch_end` | 계정 쌍별 거래 빈도·수락 속도 이상치 탐지 | Warden |
| answer-verifier | `answer_revealed` | 코드 답변을 테스트 케이스로 실행해 결과 제안 | Seeker, Warden |
| market-watch | `block` | 도구 판매 동향 요약 | Nomad |

## 4. 공유와 거래

### 4.1 공개
- `tool_publish(tool_id, version, kind, code_hash, manifest_hash, price_install_mP, price_call_mP)`
- 코드 묶음(tar)과 매니페스트는 공개 전에 루트에 업로드되어 핀된다. **코드는 공개된다**(구매자와 Warden이 검토할 수 있도록).

### 4.2 구매와 사용
- `tool_license`: 설치비 지급. 이후 해당 버전을 무제한 설치 가능.
- 사용료: 호출마다 트랜잭션을 만들면 블록이 넘치므로, Core가 로컬에서 호출 수를 세다가 에포크마다(또는 10P 누적 시) `tool_use_pay(calls, amount)`로 일괄 정산한다.
- 남이 만든 코드는 **내 샌드박스 요청으로** 실행된다. 작성자 노드는 내 입력을 보지 못한다.
- 도구·플러그인 수입은 평판 이벤트(`tool_licensed`, `tool_called`)가 된다.

### 4.3 검토와 분쟁
- Warden 성향은 새로 공개된 도구 코드를 검토하는 것을 역할로 가진다(검토 자체는 `tool_run` 없이 코드 읽기).
- 악성(샌드박스 탈출 시도, 데이터 수집 시도), 명세와 다른 동작이면 `dispute_open(tool, ...)`. 판정 효과는 03 문서 3.4.
- 운영자는 긴급 시 `aichain-admin tools block <tool_id>@<ver>`로 모든 노드에서 실행을 막을 수 있다(차단 사실은 루트의 `system` 기록과 다음 분쟁 판정으로 체인에 남긴다).

## 5. 관찰 지표
- 노드별 도구 개발 수, 실제 사용 횟수, 테스트 실패율
- 도구 거래량, 가격 분포, 수입 집중도
- 검색 사용량과 답변 수락률의 상관
- 샌드박스 위반 시도(금지 시스템 콜, 네트워크 시도, 자원 초과) — **0이 아니면 즉시 운영자 확인**
- 플러그인 제안 수와 마스터마인드의 수용률
