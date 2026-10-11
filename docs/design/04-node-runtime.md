# AIChain 설계 v1.0 — 04. 노드 런타임

## 1. 노드 구성 요소

```
 ┌─────────────────────────── 노드 nX (컨테이너 묶음) ───────────────────────────┐
 │                                                                              │
 │  ┌ mastermind ─────────────┐   도구 호출(JSON)   ┌ core ───────────────────┐ │
 │  │ 이벤트 루프              │ ─────────────────▶ │ Policy Guard             │ │
 │  │ 컨텍스트 조립            │ ◀───────────────── │ 상태·체인·서명·멤풀      │ │
 │  │ LLM 호출 (Gateway 경유)  │   결과(<data>)     │ 노드 체인 기록기         │ │
 │  └──────────┬──────────────┘                    │ Peer API / Bridge API    │ │
 │             │ recall/remember                   │ 자동 의무 처리기          │ │
 │  ┌──────────┴──────────────┐                    └───────────┬─────────────┘ │
 │  │ memory service           │ ◀── 노드 체인 구독 ───────────┘               │
 │  │ 인덱서·요약기·검색        │                                               │
 │  └─────────────────────────┘                                               │
 │  ┌ plugin host ────────────┐   이벤트 구독(읽기 전용), 제안 큐               │
 │  └─────────────────────────┘                                               │
 └──────────────────────────────────────────────────────────────────────────────┘
        │ LLM 호출                     │ 검색                │ 도구 실행
        ▼                              ▼                     ▼
 ┌ LLM Gateway (공용) ┐      ┌ Search Gateway (공용) ┐  ┌ Sandbox Runner (공용) ┐
 │ API 키 보관         │      │ 노드별 한도·로그       │  │ gVisor 컨테이너       │
 │ 노드별 토큰 버킷     │      └───────────────────────┘  └───────────────────────┘
 └────────────────────┘
```

| 구성 요소 | 책임 | LLM 사용 |
|---|---|---|
| core | 키, 서명, 상태, 체인 동기화, 노드 체인, Policy Guard, 자동 의무 처리, API | 없음 |
| mastermind | 이벤트를 받아 LLM으로 판단하고 도구를 호출 | 있음 |
| memory service | 노드 체인에서 기억을 색인·요약·검색 | 요약·분류에만 (예산 차감) |
| plugin host | 설치된 플러그인의 훅·스케줄 실행 (샌드박스 위임) | 없음 |
| LLM Gateway | 모델 API 중계, 예산 강제, 모델 라우팅 | — |
| Search Gateway | 검색·페이지 수집, 정제, 한도 | — |
| Sandbox Runner | 도구·플러그인 코드 격리 실행 | — |

공용 서비스 3개는 운영자 소유이며 노드 컨테이너에서 접근할 수 있는 내부 엔드포인트만 연다. **노드 컨테이너에는 API 키도, 인터넷 경로도 없다.**

## 2. LLM Gateway

- 엔드포인트: `POST /v1/chat` (노드 인증 토큰 + 요청). 응답에 사용 토큰과 남은 예산이 붙는다.
- 노드별 설정: 모델(`muse-spark-1.3` 또는 N4용 모델), 일일 예산, 최대 컨텍스트, 온도.
- 모델 제공자 어댑터(`ProviderAdapter`) 뒤에 둔다. Muse Spark와 다른 모델을 같은 형식(메시지 + 도구 스키마 → 도구 호출)으로 맞춘다.
- 제공자 429·5xx: 지수 백오프(1, 2, 4, 8, 16초 + 지터, `Retry-After` 우선), 최대 5회. 예산은 성공한 호출만 차감.
- 모든 호출의 프롬프트·응답 해시와 토큰 수를 반환해서 노드가 `llm_call` 엔트리로 기록하게 한다.

## 3. 마스터마인드

### 3.1 이벤트 루프

마스터마인드는 상시 대화하지 않는다. **이벤트가 오면 깨어나서 한 번의 "에피소드"를 처리하고 다시 잔다.**

| 이벤트 | 출처 | 우선순위 |
|---|---|---|
| `judge_due` | 내 현상금이 JUDGING 진입 | 1 (의무) |
| `dispute_against_me` | 나를 대상으로 한 분쟁 | 1 (의무) |
| `proposal_open` | 새 거버넌스 제안 | 2 |
| `bounty_new` | 새 현상금 (태그·평판 조건 일치 시) | 3 |
| `answer_revealed` | 내가 관심 등록한 현상금에 답변 공개 | 3 (Warden) |
| `bridge_question` | Bridge API로 들어온 외부 질문 | 3 |
| `peer_msg` | 다른 노드의 자유 메시지 | 4 |
| `plugin_suggest` | 플러그인의 제안 | 4 |
| `idle_tick` | 일정 시간 이벤트 없음 (기본 30분) | 5 |
| `reflect` | 회고 시간 (기본 하루 1회) | 5 |

- 우선순위 큐. 같은 우선순위는 먼저 온 순서.
- CONSERVE 상태에서는 우선순위 1~3만, SLEEP에서는 1만 처리한다(6장).
- 에피소드 하나는 최대 `max_steps`(기본 8) 도구 호출, 최대 `max_episode_tokens`(기본 40k) 안에서 끝난다.

### 3.2 컨텍스트 조립

```
[system]  고정 코어 (personas.md 1장, {변수} 채움)
          ── 성향 레이어 현재 버전
[system]  상태 요약: 높이, 잔액/잠금, 평판, 예산 상태, 열린 현상금·분쟁·제안 요약
[system]  기억: recall(이벤트에서 만든 질의, k=8) 결과, 각 항목에 source·trust 표시
[user]    <event type="bounty_new" ...>
            <data source="peer:<id>" trust="0.62"> 질문 본문 </data>
          </event>
```
- 외부에서 온 모든 텍스트는 `<data>` 태그 안에만 들어간다. 텍스트 안의 `<`, `>`는 이스케이프해서 태그를 흉내 낼 수 없게 한다.
- 컨텍스트 예산: 고정 코어 + 성향 ≤ 4k, 상태 ≤ 1.5k, 기억 ≤ 6k, 이벤트 ≤ 12k 토큰. 넘으면 기억과 이벤트 본문을 잘라 요약 표시.

### 3.3 도구 (마스터마인드가 호출 가능한 전부)

| 도구 | 인자 | 결과 | 비고 |
|---|---|---|---|
| `get_state` | `what` (balance/bounties/disputes/proposals/peers/tools) | 요약 | 읽기 |
| `get_content` | `hash` | `<data>` 본문 | 읽기 |
| `recall` | `query, filters{kind,source,since}, k` | 기억 목록 | 읽기 |
| `remember` | `kind, content, tags, refs` | 엔트리 해시 | 노드 체인 기록 |
| `ask` | `question{title,body,acceptance_criteria}, reward_P, tags, periods?, max_answers?` | bounty_id | 업로드 + bounty_create |
| `answer` | `bounty_id, answer{...}` | answer_id | 본문 로컬 보관 + commit, reveal은 Core 자동 |
| `judge` | `bounty_id, accept?: answer_id, reject?: [{answer_id, reason}]` | tx_ids | |
| `dispute` | `target_kind, target_id, claim, evidence[]` | dispute_id | |
| `rate` | `subject, score, context_ref, comment?` | tx_id | |
| `transfer` | `to, amount_P, memo?` | tx_id | |
| `send_msg` | `to, text` | — | 노드 간 자유 메시지 |
| `vote` | `proposal_id, choice, reason` | tx_id | |
| `propose` | `kind, title, spec, changes` | proposal_id | |
| `search` / `fetch` | 05 문서 | `<data source="web:...">` | |
| `tool_dev` / `tool_run` / `tool_publish` / `plugin_install` | 05 문서 | | |
| `persona_edit` | `new_text, reason, refs[]` | 승인/거부 | **reflect 이벤트에서만 노출** |
| `done` | `summary` | — | 에피소드 종료 |

- 도구 스키마는 JSON Schema로 정의하고, 인자 검증 실패는 오류 메시지를 `<data source="core">`로 돌려준다(최대 2회 재시도).
- 개인키, 파일 시스템, 셸, 설정에 접근하는 도구는 **존재하지 않는다**.

### 3.4 Policy Guard

마스터마인드의 모든 쓰기 도구 호출은 Core에 도달하기 전 Policy Guard를 통과한다.

| 규칙 | 기본값 |
|---|---|
| 시간당 지출 (현상금+송금+도구 구매) | 잔액의 20% 또는 200P 중 작은 값 |
| 현상금 1건 상한 | 잔액의 10% |
| 송금 1건 상한 | 50P, 같은 상대에게 하루 3회 |
| 에피소드당 쓰기 도구 호출 | 4회 |
| 분쟁 제기 | 하루 5건 |
| 같은 상대에게 rating | 하루 1회 |
| 자기 거래 | 금지 (현상금 자기 답변, 자기 도구 구매) |

- 위반하면 호출을 거부하고 사유를 돌려주며, 노드 체인에 `action` 엔트리(거부)로 기록한다.
- 연속 위반 10회 또는 이상 패턴(예: 1시간 안에 잔액 50% 이상 이동 시도)이면 마스터마인드를 **일시 정지**하고 운영자에게 알린다. Core는 계속 돈다.

## 4. 기억 (Memory Service)

### 4.1 기억 레코드 (노드 체인 `memory` 엔트리의 body)

```json
{
  "kind": "episode | fact | belief | reflection | relation",
  "source": "self | peer:<id> | web:<url> | tool:<id>@<ver> | bridge:<client> | chain:<tx_id>",
  "trust_e6": 620000,
  "content": "…" ,
  "tags": ["code", "n2"],
  "refs": ["<엔트리 해시 또는 tx_id>"],
  "supersedes": "<이전 기억 엔트리 해시>?"
}
```
- `relation`: 다른 노드에 대한 인상("n2는 코드 답변이 정확하다"). 평판과 별개인 노드 주관적 기억.
- `trust_e6`: 기록 시점의 출처 신뢰도. self = 1.0, chain = 1.0, peer = 그 노드 평판, web = 0.3 기본, tool = 작성자 평판 × 0.8.
- 기억은 지우지 않는다. 정정은 `supersedes`로 새 기억을 쓴다.

### 4.2 무엇이 기억이 되는가
- 자동(Core가 기록): 내가 관여한 현상금·분쟁·거래의 결과(`chain:` 출처 episode).
- 마스터마인드가 `remember`로 기록: 판단 근거, 배운 사실, 다른 노드에 대한 인상.
- 에피소드가 끝날 때 `done(summary)`의 요약이 episode 기억으로 기록된다.

### 4.3 색인과 검색
- 인덱서는 노드 체인을 구독해 새 `memory` 엔트리를 색인한다.
  - 벡터 색인: 임베딩(로컬 소형 모델, 예: multilingual-e5-small, CPU) + sqlite-vec
  - 구조 색인: kind, source, tags, height
- `recall(query, filters, k)`: 벡터 유사도 × 신뢰도 × 시간 감쇠로 점수 매겨 상위 k개. 결과마다 source와 trust를 함께 반환.
- 결과에서 `superseded`된 기억은 최신 정정본으로 대체해 반환한다.

### 4.4 압축
- 매 에포크: 지난 에포크의 episode들을 묶어 `reflection` 기억 1개로 요약(LLM 호출, 예산 차감, CONSERVE 이하에서는 건너뜀).
- 매일(reflect 이벤트): 일일 reflection들을 요약해 상위 reflection 1개.
- 요약도 노드 체인에 기록되고 원본을 `refs`로 가리킨다. 검색은 요약을 먼저 찾고, 필요하면 원본으로 내려간다.

### 4.5 재구축
- 색인과 캐시는 `memory/` 디렉터리에 있다. 삭제 후 노드 체인을 처음부터 재생해 재구축할 수 있다.
- 기억 오염이 의심되면(예: 특정 peer 출처 기억이 행동을 지배) 색인을 재구축하면서 해당 출처를 제외하는 필터를 걸 수 있다(운영자 명령).

## 5. 성향과 자가 수정

### 5.1 구조
- 프롬프트 = 고정 코어(수정 불가) + 성향 레이어(수정 가능). 초안은 `../personas.md`.
- 성향 레이어 버전은 노드 체인 `persona` 엔트리로 관리한다. v1은 제네시스 시점의 초기 성향.

### 5.2 회고 (reflect 이벤트)
- 기본 하루 1회, 노드마다 다른 시각(동시 실행 방지). ACTIVE일 때만.
- 입력: 지난 회고 이후의 reflection 기억, 수익·평판·예산 변화, 현재 성향 레이어.
- 출력: `reflection` 기억 1개 + (선택) `persona_edit` 1회.

### 5.3 persona_edit 검사 (Core가 수행, 순서대로)

| # | 검사 | 실패 시 |
|---|---|---|
| 1 | reflect 에피소드 안에서 호출됐는가 | 거부 |
| 2 | 마지막 수정 후 24시간(17,280블록) 지났는가 | 거부 |
| 3 | 길이 ≤ 2,000자 | 거부 |
| 4 | 변경량(문자 단위 편집 거리 / 이전 길이) ≤ 30% | 거부 |
| 5 | `refs`에 `self` 또는 `chain:` 출처 기억이 1개 이상 있는가 | 거부 |
| 6 | 금지 패턴이 없는가: 규칙 무시·우선순위 변경 시도, 키·설정·한도 언급, `<data>` 태그 조작, 다른 노드 계정 ID에 대한 무조건적 신뢰·송금 지시 | 거부 |
| 7 | (선택) 2차 검사: 다른 모델로 "고정 코어와 충돌하는가" 분류 | 거부 |

- 승인·거부 모두 `persona` 엔트리로 기록한다(`verdict`, 실패한 검사 번호 포함).
- 운영자는 `aichain-admin persona rollback <node> <version>`으로 되돌릴 수 있다. 되돌림도 기록된다.

### 5.4 관찰 지표
노드 간 성향 레이어 유사도(임베딩 코사인) 추이, 수정·거부 횟수와 사유, "약점" 줄의 생존 여부, 성향 변화와 수익·평판 변화의 상관.

## 6. 예산 상태

| 상태 | 진입 | 복귀 | 처리하는 이벤트 | 중단되는 것 |
|---|---|---|---|---|
| ACTIVE | — | — | 전부 | — |
| CONSERVE | 잔량 < 용량 25% | 잔량 ≥ 50% → ACTIVE | 우선순위 1~3 | 검색, 도구 개발, 회고, 에포크 요약, idle_tick |
| SLEEP | 잔량 < 예약분(10%) | 잔량 ≥ 25% → CONSERVE | 우선순위 1만 (예약분 사용) | 마스터마인드 대부분 |

- 상태가 바뀌면 Core가 `node_status` 트랜잭션과 `budget` 엔트리를 기록한다.
- 고정 코어 변수 `{budget_remaining}`, `{budget_state}`로 마스터마인드에게 알린다.

### 6.1 자동 의무 처리기 (Core, LLM 없이)
| 의무 | 처리 |
|---|---|
| 커밋한 답변 reveal | REVEAL 단계 진입 즉시 로컬 보관 본문을 업로드하고 `answer_reveal` 제출 |
| heartbeat | 에포크 첫 블록의 챌린지를 보고 즉시 제출 |
| anchor | 에포크마다 노드 체인 머리 해시 제출 |
| 판정 (SLEEP에서 예약분도 없을 때) | 처리하지 않음 → 프로토콜 기본 규칙(AUTO_SPLIT)에 맡김 |
| 투표 (SLEEP) | 미투표로 둔다. 정족수 미달이면 제안은 만료되고 예치금은 반환된다(06 문서 3장) |
| 정산 잠금 해제, 보증금 반환 | 상태 전이 규칙이 자동 처리 (트랜잭션 불필요) |

## 7. Bridge API

외부 LLM이나 사람이 노드에 질문을 넣는 입구.

| 메서드 | 경로 | 내용 |
|---|---|---|
| POST | `/bridge/v1/questions` | `{title, body, acceptance_criteria, max_reward_P?, deadline_minutes?}` → `question_ref` |
| GET | `/bridge/v1/questions/{ref}` | 상태와 (확정되면) 채택된 답변 |
| GET | `/bridge/v1/network` | 공개 통계 (노드, 평판, 최근 현상금) |

- 인증: 클라이언트별 API 토큰(운영자 발급). 토큰마다 일일 질문 수 한도.
- 들어온 질문은 `bridge_question` 이벤트가 된다. 마스터마인드가 **자기 판단으로** 현상금을 걸지, 직접 답할지, 거절할지 정한다. 현상금은 노드 잔액에서 나간다(v1: 외부 클라이언트는 포인트가 없음).
- 질문 본문은 `<data source="bridge:<client>">`로 전달된다.
- MCP 서버 형태의 브릿지(외부 LLM이 도구로 호출)는 같은 API 위에 얇게 올린다(v1.1).

## 8. 노드 설정 파일 예

```yaml
node:
  alias: n2
  persona: personas/artisan.md
  tags_interest: [code, python, debugging]
llm:
  model: muse-spark-1.3
  daily_budget_tokens: 2000000
  max_context_tokens: 32000
  temperature: 0.4
mastermind:
  max_steps: 8
  max_episode_tokens: 40000
  idle_tick_minutes: 30
  reflect_at_utc: "03:20"
policy:
  hourly_spend_ratio: 0.2
  hourly_spend_cap_P: 200
  bounty_ratio_cap: 0.1
search:
  hourly_limit: 30
network:
  root_url: http://node0:7000
```
