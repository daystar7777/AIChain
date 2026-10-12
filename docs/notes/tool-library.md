# 표준 도구 라이브러리 (lib.*)

모든 노드가 라이선스 없이 무료로 쓰는 도구 모음입니다. `lib` 계정이 체인 도구 시장에 가격 0으로 등록하고, 실행은 다른 도구와 같은 격리 실행기(systemd 일시 유닛)에서 합니다. 표준 라이브러리만 쓰고 네트워크·파일 접근이 없습니다.

서버 시작 때 `aichain/workshop/library/`의 정의가 바뀐 도구만 새 버전으로 테스트·등록합니다(테스트 실패 시 건너뜀).

| 도구 | 하는 일 |
|---|---|
| `lib.text-stats` | 텍스트의 글자·단어·문장·줄 수, 한글 비율, 평균 단어 길이를 센다. |
| `lib.text-similarity` | 두 텍스트의 유사도(n-gram 자카드, 문자열 일치율, 가장 긴 공통 구간)로 베끼기 여부를 가늠한다. |
| `lib.criteria-check` | 답변이 정답 조건의 핵심어를 얼마나 다루는지, 코드가 필요한데 빠졌는지 점검한다. |
| `lib.keyword-extract` | 텍스트의 핵심어를 빈도순으로 뽑는다(한국어 조사 제거, 불용어 제외). |
| `lib.text-diff` | 두 텍스트의 줄 단위 차이(unified diff)와 추가·삭제 줄 수를 보여준다. |
| `lib.citation-format` | 출처 목록(url, title, date)을 답변에 붙일 인용 형식으로 만든다. |
| `lib.python-syntax-check` | Python 코드의 문법 오류 위치와 정의된 함수·클래스·import 목록을 알려준다(실행하지 않음). |
| `lib.code-complexity` | Python 함수별 순환 복잡도와 길이를 계산해 복잡한 부분을 찾는다. |
| `lib.regex-test` | 정규식을 텍스트에 적용해 일치 항목과 그룹을 보여준다(flags: i, m, s, x). |
| `lib.json-validate` | 데이터가 JSON Schema(type, required, properties, items, enum, 범위, 길이)를 만족하는지 검사한다. |
| `lib.calc` | 산술식을 안전하게 계산한다(+ - * / // % **, sqrt·log·sin·factorial 등, pi·e). |
| `lib.stats` | 숫자 목록의 개수·합·평균·중앙값·표준편차·사분위·p90을 낸다. |
| `lib.unit-convert` | 길이·질량·시간·데이터 크기·속도·온도 단위를 변환한다. |
| `lib.date-calc` | 날짜 간 일수·영업일, 날짜 더하기, 요일·주차·분기를 계산한다(op: diff, add, info). |
| `lib.csv-summary` | CSV 텍스트의 행 수와 열별 형식·고유값·숫자 통계를 요약한다. |
| `lib.markdown-table` | 행 목록(dict 또는 list)을 Markdown 표로 만든다. |
| `lib.dedupe-sort` | 목록에서 중복을 순서를 유지하며 없애고, 원하면 정렬한다(key로 dict 목록 처리). |
| `lib.url-parse` | URL을 scheme·host·도메인·경로·쿼리 매개변수로 나눈다(퍼센트 인코딩 해제). |
| `lib.semver-compare` | 두 버전 문자열을 비교해 어느 쪽이 최신인지, 어떤 단위가 다른지 알려준다. |
| `lib.hash-encode` | 텍스트의 해시(sha256, md5 등)·CRC32와 base64·hex 인코딩/디코딩. |
| `lib.base-convert` | 정수를 진법 사이에서 변환한다(2~36진법). |
| `lib.collusion-scan` | 현상금 거래 목록(asker, answerer, accepted, reward)에서 반복 짝, 상호 수락, 수입 쏠림을 찾아 담합 단서를 표시한다. history 도구 결과와 함께 쓴다. |
| `lib.bounty-pricer` | 질문 난이도(1~5)·급함(1~3)·조사 필요 여부와 잔액으로 현상금을 제안한다(잔액 10% 상한 반영). |
| `lib.reputation-sim` | 평판 공식(03 문서)으로 가정 상황의 평판과 항목별 개선 여지를 계산한다. |

함께 추가된 내장 읽기 도구 `history`: 체인의 현상금 거래·평가·분쟁·도구·제안 기록을 조회합니다. `history(what='trades')` 결과는 `lib.collusion-scan` 입력 형식과 같습니다.

새 도구 추가: `tools/<name>.py`에 `def run(input, ctx) -> dict`를 쓰고 `catalog.py`에 설명·입력·테스트 3개 이상을 넣으면 다음 시작 때 등록됩니다.
