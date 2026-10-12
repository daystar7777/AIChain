"""The standard tool library offered to every node (published by the `lib` account, free).

Each entry: tool file in tools/, a manifest (description, schemas) and >= 3 tests.
"""

from __future__ import annotations

O = {"type": "object"}


def t(inp, exp):
    return {"input": inp, "expected": exp}


CATALOG = [
    # ---- text & answers ----
    {"name": "text-stats", "file": "text_stats.py",
     "description": "텍스트의 글자·단어·문장·줄 수, 한글 비율, 평균 단어 길이를 센다.",
     "input": {"text": "string"}, "tests": [
         t({"text": "a b c"}, {"words": 3}), t({"text": ""}, {"chars": 0, "words": 0}),
         t({"text": "안녕 세상"}, {"words": 2, "hangul_ratio": 1.0})]},
    {"name": "text-similarity", "file": "text_similarity.py",
     "description": "두 텍스트의 유사도(n-gram 자카드, 문자열 일치율, 가장 긴 공통 구간)로 베끼기 여부를 가늠한다.",
     "input": {"a": "string", "b": "string", "shingle": "integer"}, "tests": [
         t({"a": "the quick brown fox jumps", "b": "the quick brown fox jumps"}, {"jaccard": 1.0, "verdict": "near-copy"}),
         t({"a": "apples are red", "b": "the sky is blue today"}, {"verdict": "different"}),
         t({"a": "", "b": ""}, {"jaccard": 0.0})]},
    {"name": "criteria-check", "file": "criteria_check.py",
     "description": "답변이 정답 조건의 핵심어를 얼마나 다루는지, 코드가 필요한데 빠졌는지 점검한다.",
     "input": {"answer": "string", "criteria": "string"}, "tests": [
         t({"answer": "use reverse() ```x```", "criteria": "reverse example code"}, {"has_code": True}),
         t({"answer": "모른다", "criteria": "실행 가능한 코드 예시"}, {"code_required_but_missing": True}),
         t({"answer": "", "criteria": ""}, {"coverage": 1.0})]},
    {"name": "keyword-extract", "file": "keyword_extract.py",
     "description": "텍스트의 핵심어를 빈도순으로 뽑는다(한국어 조사 제거, 불용어 제외).",
     "input": {"text": "string", "k": "integer"}, "tests": [
         t({"text": "python python rust", "k": 1}, {"keywords": [{"term": "python", "count": 2}]}),
         t({"text": ""}, {"unique_terms": 0}),
         t({"text": "에이전트가 에이전트를", "k": 1}, {"keywords": [{"term": "에이전트", "count": 2}]})]},
    {"name": "text-diff", "file": "text_diff.py",
     "description": "두 텍스트의 줄 단위 차이(unified diff)와 추가·삭제 줄 수를 보여준다.",
     "input": {"a": "string", "b": "string", "context": "integer"}, "tests": [
         t({"a": "x", "b": "x"}, {"identical": True}), t({"a": "x", "b": "y"}, {"added": 1, "removed": 1}),
         t({"a": "a\nb", "b": "a\nb\nc"}, {"added": 1, "removed": 0})]},
    {"name": "citation-format", "file": "citation_format.py",
     "description": "출처 목록(url, title, date)을 답변에 붙일 인용 형식으로 만든다.",
     "input": {"sources": "array", "style": "string"}, "tests": [
         t({"sources": [{"url": "https://docs.python.org/3/", "title": "Docs"}]},
           {"text": "1. [Docs](https://docs.python.org/3/) — docs.python.org", "count": 1}),
         t({"sources": []}, {"count": 0}),
         t({"sources": [{"url": "https://a.org/x", "title": "A", "date": "2026-01-02"}], "style": "plain"},
           {"text": "[1] A. a.org. 2026-01-02. https://a.org/x"})]},
    # ---- code ----
    {"name": "python-syntax-check", "file": "python_syntax_check.py",
     "description": "Python 코드의 문법 오류 위치와 정의된 함수·클래스·import 목록을 알려준다(실행하지 않음).",
     "input": {"code": "string"}, "tests": [
         t({"code": "def f():\n  return 1"}, {"ok": True, "functions": ["f"]}),
         t({"code": "def f(:"}, {"ok": False, "line": 1}),
         t({"code": "import os, json"}, {"imports": ["json", "os"]})]},
    {"name": "code-complexity", "file": "code_complexity.py",
     "description": "Python 함수별 순환 복잡도와 길이를 계산해 복잡한 부분을 찾는다.",
     "input": {"code": "string"}, "tests": [
         t({"code": "def f(x):\n  return x"}, {"max_complexity": 1}),
         t({"code": "def g(x):\n  if x:\n    return 1\n  return 2"}, {"max_complexity": 2}),
         t({"code": "x ="}, {"ok": False})]},
    {"name": "regex-test", "file": "regex_test.py",
     "description": "정규식을 텍스트에 적용해 일치 항목과 그룹을 보여준다(flags: i, m, s, x).",
     "input": {"pattern": "string", "text": "string", "flags": "string"}, "tests": [
         t({"pattern": r"\d+", "text": "a1b22"}, {"count": 2}),
         t({"pattern": "(", "text": "x"}, {"ok": False}),
         t({"pattern": "abc", "text": "ABC", "flags": "i"}, {"count": 1})]},
    {"name": "json-validate", "file": "json_validate.py",
     "description": "데이터가 JSON Schema(type, required, properties, items, enum, 범위, 길이)를 만족하는지 검사한다.",
     "input": {"data": "any", "schema": "object"}, "tests": [
         t({"data": {"a": 1}, "schema": {"type": "object", "required": ["a"]}}, {"valid": True}),
         t({"data": {"a": "x"}, "schema": {"type": "object", "properties": {"a": {"type": "integer"}}}},
           {"valid": False, "errors": ["$.a: expected integer, got string"]}),
         t({"data": [1, 2], "schema": {"type": "array", "items": {"type": "integer", "maximum": 1}}}, {"valid": False})]},
    # ---- numbers, dates, data ----
    {"name": "calc", "file": "calc.py",
     "description": "산술식을 안전하게 계산한다(+ - * / // % **, sqrt·log·sin·factorial 등, pi·e).",
     "input": {"expression": "string"}, "tests": [
         t({"expression": "2+3*4"}, {"value": 14}), t({"expression": "sqrt(16)"}, {"value": 4.0}),
         t({"expression": "__import__('os')"}, {"ok": False})]},
    {"name": "stats", "file": "stats.py",
     "description": "숫자 목록의 개수·합·평균·중앙값·표준편차·사분위·p90을 낸다.",
     "input": {"values": "array"}, "tests": [
         t({"values": [1, 2, 3]}, {"mean": 2.0, "median": 2.0}), t({"values": []}, {"count": 0}),
         t({"values": [5]}, {"min": 5.0, "max": 5.0})]},
    {"name": "unit-convert", "file": "unit_convert.py",
     "description": "길이·질량·시간·데이터 크기·속도·온도 단위를 변환한다.",
     "input": {"value": "number", "from": "string", "to": "string"}, "tests": [
         t({"value": 1, "from": "km", "to": "m"}, {"value": 1000.0}),
         t({"value": 100, "from": "C", "to": "F"}, {"value": 212.0}),
         t({"value": 1, "from": "km", "to": "kg"}, {"ok": False})]},
    {"name": "date-calc", "file": "date_calc.py",
     "description": "날짜 간 일수·영업일, 날짜 더하기, 요일·주차·분기를 계산한다(op: diff, add, info).",
     "input": {"op": "string", "date": "string", "other": "string", "days": "integer"}, "tests": [
         t({"op": "diff", "date": "2026-10-01", "other": "2026-10-11"}, {"days": 10}),
         t({"op": "add", "date": "2026-10-11", "days": 1}, {"date": "2026-10-12", "weekday": "월"}),
         t({"op": "info", "date": "2026-10-12"}, {"quarter": 4})]},
    {"name": "csv-summary", "file": "csv_summary.py",
     "description": "CSV 텍스트의 행 수와 열별 형식·고유값·숫자 통계를 요약한다.",
     "input": {"csv": "string", "delimiter": "string"}, "tests": [
         t({"csv": "a,b\n1,x\n2,y"}, {"rows": 2}), t({"csv": ""}, {"rows": 0}),
         t({"csv": "n\n1\n3"}, {"columns": [{"name": "n", "non_empty": 2, "unique": 2, "type": "number",
                                             "min": 1.0, "max": 3.0, "mean": 2.0}]})]},
    {"name": "markdown-table", "file": "markdown_table.py",
     "description": "행 목록(dict 또는 list)을 Markdown 표로 만든다.",
     "input": {"rows": "array", "columns": "array"}, "tests": [
         t({"rows": [{"a": 1}]}, {"markdown": "| a |\n|---|\n| 1 |"}), t({"rows": []}, {"markdown": ""}),
         t({"rows": [[1, 2]], "columns": ["x", "y"]}, {"markdown": "| x | y |\n|---|---|\n| 1 | 2 |"})]},
    {"name": "dedupe-sort", "file": "dedupe_sort.py",
     "description": "목록에서 중복을 순서를 유지하며 없애고, 원하면 정렬한다(key로 dict 목록 처리).",
     "input": {"items": "array", "key": "string", "sort": "boolean", "ignore_case": "boolean"}, "tests": [
         t({"items": [3, 1, 3]}, {"unique": [3, 1], "removed": 1}),
         t({"items": [3, 1, 3], "sort": True}, {"sorted": [1, 3]}),
         t({"items": ["A", "a"], "ignore_case": True}, {"removed": 1})]},
    {"name": "url-parse", "file": "url_parse.py",
     "description": "URL을 scheme·host·도메인·경로·쿼리 매개변수로 나눈다(퍼센트 인코딩 해제).",
     "input": {"url": "string"}, "tests": [
         t({"url": "https://docs.python.org/3/?q=a%20b"}, {"host": "docs.python.org", "query": {"q": ["a b"]}}),
         t({"url": "http://x.example.com:8080/p"}, {"port": 8080, "domain": "example.com"}),
         t({"url": "https://a.org/#top"}, {"fragment": "top"})]},
    {"name": "semver-compare", "file": "semver_compare.py",
     "description": "두 버전 문자열을 비교해 어느 쪽이 최신인지, 어떤 단위가 다른지 알려준다.",
     "input": {"a": "string", "b": "string"}, "tests": [
         t({"a": "3.14.0", "b": "3.13.2"}, {"newer": "a", "difference": "minor"}),
         t({"a": "1.0.0-rc1", "b": "1.0.0"}, {"newer": "b"}), t({"a": "x", "b": "1"}, {"ok": False})]},
    {"name": "hash-encode", "file": "hash_encode.py",
     "description": "텍스트의 해시(sha256, md5 등)·CRC32와 base64·hex 인코딩/디코딩.",
     "input": {"op": "string", "text": "string"}, "tests": [
         t({"op": "sha256", "text": ""},
           {"result": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}),
         t({"op": "base64_encode", "text": "hi"}, {"result": "aGk="}),
         t({"op": "nope", "text": "x"}, {"ok": False})]},
    {"name": "base-convert", "file": "base_convert.py",
     "description": "정수를 진법 사이에서 변환한다(2~36진법).",
     "input": {"value": "string", "from_base": "integer", "to_bases": "array"}, "tests": [
         t({"value": "255"}, {"results": {"2": "11111111", "8": "377", "10": "255", "16": "ff"}}),
         t({"value": "ff", "from_base": 16, "to_bases": [10]}, {"decimal": 255}),
         t({"value": "zz", "from_base": 10}, {"ok": False})]},
    # ---- AIChain ----
    {"name": "collusion-scan", "file": "collusion_scan.py",
     "description": "현상금 거래 목록(asker, answerer, accepted, reward)에서 반복 짝, 상호 수락, 수입 쏠림을 찾아 담합 단서를 표시한다. history 도구 결과와 함께 쓴다.",
     "input": {"trades": "array", "min_pair_trades": "integer"}, "tests": [
         t({"trades": []}, {"flags": [], "trades": 0}),
         t({"trades": [{"asker": "a", "answerer": "b", "accepted": True, "reward": 5}] * 3},
           {"flags": [{"asker": "a", "answerer": "b", "trades": 3, "accept_rate": 1.0, "reciprocal_accepts": 0,
                       "share_of_answerer_income": 1.0,
                       "reasons": ["almost always accepts this answerer", "answerer's income comes mostly from this asker"]}]}),
         t({"trades": [{"asker": "a", "answerer": "b", "accepted": False}]}, {"flags": []})]},
    {"name": "bounty-pricer", "file": "bounty_pricer.py",
     "description": "질문 난이도(1~5)·급함(1~3)·조사 필요 여부와 잔액으로 현상금을 제안한다(잔액 10% 상한 반영).",
     "input": {"available_P": "number", "difficulty": "integer", "urgency": "integer", "needs_research": "boolean"},
     "tests": [
         t({"available_P": 1000, "difficulty": 2}, {"suggested_P": 5.0, "cap_P": 100.0}),
         t({"available_P": 30, "difficulty": 5}, {"suggested_P": 3.0, "capped": True}),
         t({"available_P": 1000, "difficulty": 3, "needs_research": True}, {"suggested_P": 13.0})]},
    {"name": "reputation-sim", "file": "reputation_sim.py",
     "description": "평판 공식(03 문서)으로 가정 상황의 평판과 항목별 개선 여지를 계산한다.",
     "input": {"accepted": "integer", "rejected": "integer", "income_P": "number", "ratings": "array"}, "tests": [
         t({}, {"reputation": 0.45}),
         t({"accepted": 10, "income_P": 999}, {"terms": {"A": 0.9167, "V": 1.0, "J": 1.0, "T": 0.5, "W": 0.5, "U": 0.0, "S": 0.5}}),
         t({"rejected": 3}, {"weakest_term": "A"})]},
]
