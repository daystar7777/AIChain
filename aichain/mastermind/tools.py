"""Tool schema v0: what a Mastermind may call (design doc 04, 3.3).

This prototype exposes the read, memory and bounty tools plus `done`. Argument models
are pydantic; their JSON Schema is what the model sees.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from aichain.gateways.llm.base import ToolSpec
from aichain.workshop.manifest import ToolTest, ToolManifest


class GetStateArgs(BaseModel):
    what: Literal["balance", "bounties", "my_bounties"] = Field(
        description="balance: 내 잔액·잠금·예산. bounties: 열린 현상금 목록. my_bounties: 내가 만들었거나 답한 현상금."
    )


class GetContentArgs(BaseModel):
    hash: str = Field(description="콘텐츠 해시 (질문·답변 본문)")


class RecallArgs(BaseModel):
    query: str = Field(description="찾고 싶은 기억을 설명하는 검색어")
    k: int = Field(default=5, ge=1, le=8)


class RememberArgs(BaseModel):
    kind: Literal["fact", "belief", "relation", "episode"]
    content: str = Field(max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=8)
    refs: list[str] = Field(default_factory=list, max_length=8)
    source_url: str | None = Field(
        default=None, description="웹에서 읽은 사실이면 그 페이지 URL. 이번 실행에서 fetch한 URL만 허용.")


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="검색어")
    k: int = Field(default=5, ge=1, le=8)


class FetchArgs(BaseModel):
    url: str = Field(max_length=2000, description="읽을 페이지 URL (http/https)")
    max_chars: int = Field(default=8000, ge=500, le=20000)


class NoteFact(BaseModel):
    text: str = Field(max_length=600)
    source_url: str = Field(max_length=2000, description="이 사실의 출처. fetch한 URL만 허용.")


class NoteArgs(BaseModel):
    topic: str = Field(max_length=120, description="관심사 주제 이름 (같은 주제면 이전 노트를 대체)")
    summary: str = Field(max_length=2000, description="이 주제에 대해 지금까지 알게 된 것의 요약")
    facts: list[NoteFact] = Field(default_factory=list, max_length=12)
    confidence: float = Field(ge=0, le=1)
    open_questions: list[str] = Field(default_factory=list, max_length=5,
                                      description="다음에 더 알아볼 것")


class Question(BaseModel):
    title: str = Field(max_length=200)
    body: str = Field(max_length=8000)
    acceptance_criteria: str = Field(
        max_length=2000, description="답이 맞았다고 판정할 조건. 판정과 분쟁의 기준이 된다."
    )


class AskArgs(BaseModel):
    question: Question
    reward_P: float = Field(ge=1, description="현상금(P). 즉시 잠긴다.")
    tags: list[str] = Field(default_factory=list, max_length=5)
    max_answers: int | None = Field(default=None, ge=1, le=20)


class AnswerBody(BaseModel):
    answer: str = Field(max_length=8000)
    confidence: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list, max_length=10)
    verification: str = Field(
        default="", max_length=2000, description="정답 조건을 충족함을 확인하는 방법"
    )


class AnswerArgs(BaseModel):
    bounty_id: str
    answer: AnswerBody


class Rejection(BaseModel):
    answer_id: str
    reason: str = Field(max_length=1000)


class JudgeArgs(BaseModel):
    bounty_id: str
    accept: str | None = Field(default=None, description="수락할 answer_id (하나만)")
    reject: list[Rejection] = Field(
        default_factory=list, description="정답 조건을 충족하지 못한 답변만. 수락과 함께 쓸 필요 없음.")


class ToolDevArgs(BaseModel):
    manifest: ToolManifest
    code: str = Field(max_length=200_000, description="Python 코드. 최상위에 def run(input, ctx) -> dict")
    tests: list[ToolTest] = Field(min_length=3, max_length=20)
    reason: str = Field(max_length=1000, description="이 도구가 왜 필요한지")


class ToolListArgs(BaseModel):
    tool_id: str | None = Field(default=None, description="비우면 내 도구 목록, 지정하면 상세(코드·테스트 결과)")
    version: int | None = None


class ToolActivateArgs(BaseModel):
    tool_id: str
    version: int
    mode: Literal["run", "update"] = Field(
        description="run: 첫 활성화·재시작·current로 롤백. update: 다른 버전으로 전환")


class ToolRunArgs(BaseModel):
    tool_id: str
    input: dict = Field(default_factory=dict)


class DoneArgs(BaseModel):
    summary: str = Field(max_length=1000, description="이번 에피소드에서 한 일과 이유, 한두 문장")


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "get_state": (GetStateArgs, "현재 상태를 조회한다."),
    "get_content": (GetContentArgs, "해시로 콘텐츠 본문을 읽는다. 결과는 <data>다."),
    "recall": (RecallArgs, "내 기억에서 관련 항목을 찾는다. 각 항목에 source와 trust가 붙는다."),
    "remember": (RememberArgs, "판단 근거, 배운 사실, 다른 노드에 대한 인상을 내 노드 체인에 기록한다."),
    "ask": (AskArgs, "질문을 올리고 현상금을 건다. 현상금은 즉시 잠긴다."),
    "answer": (AnswerArgs, "열린 현상금에 답한다. 커밋 보증금 0.5P가 잠기고, 공개는 Core가 자동으로 한다."),
    "judge": (JudgeArgs, "내 현상금의 답변을 판정한다. 하나를 수락하면 현상금이 그 답변자에게 가고 판정이 끝난다. "
              "나머지 답은 거절할 필요가 없다. reject는 정답 조건을 충족하지 못한 답에만 쓴다 "
              "(정답을 거절하면 상대 평판이 깎이고 분쟁 대상이 된다)."),
    "search": (SearchArgs, "헤드리스 브라우저로 웹을 검색한다. 결과(제목, URL, 요약)는 <data>다."),
    "fetch": (FetchArgs, "웹 페이지를 열어 본문 텍스트를 읽는다. 결과는 <data source=\"web:URL\">다."),
    "note": (NoteArgs, "관심사 주제의 지식 노트를 쓰거나 갱신한다. 같은 주제의 이전 노트는 새 노트로 대체되고 기록은 남는다."),
    "tool_dev": (ToolDevArgs, "새 도구(또는 기존 도구의 새 버전)를 정의한다. 정적 검사와 테스트를 거쳐 "
                              "불변 버전으로 기록된다. 실패하면 이유와 고칠 방법이 돌아온다."),
    "tool_list": (ToolListArgs, "내 도구 목록, 또는 한 도구의 코드·테스트 결과를 본다."),
    "tool_activate": (ToolActivateArgs, "테스트를 통과한 버전을 활성화한다. 실패한 업데이트는 current를 바꾸지 않는다."),
    "tool_run": (ToolRunArgs, "활성화된 내 도구를 실행한다. 결과는 <data source=\"tool:...\">다."),
    "done": (DoneArgs, "에피소드를 끝낸다."),
}

WEB_TOOLS = {"search", "fetch", "note"}
WORKSHOP_TOOLS = {"tool_dev", "tool_list", "tool_activate", "tool_run"}

WRITE_TOOLS = {"ask", "answer", "judge"}


def _inline(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs and drop pydantic titles: some providers reject $ref."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def tool_specs(names: list[str] | None = None) -> list[ToolSpec]:
    names = names or list(TOOLS)
    return [
        ToolSpec(name=n, description=TOOLS[n][1], parameters=_inline(TOOLS[n][0].model_json_schema()))
        for n in names
    ]
