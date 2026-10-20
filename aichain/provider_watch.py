"""Tell the manager when the LLM provider keeps failing (e.g. an expired API key).

Episodes end with stopped="provider_error" when the model cannot be reached. One of those is
noise; an authentication error, or several in a row, means every node has stopped thinking while
blocks keep coming, which nobody would notice. The watch sends one Telegram message (manager
outbox) then, repeats it at most every few hours while it lasts, and one when calls work again.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

AUTH = re.compile(r"\b(401|403)\b|api[_ ]key|authenticat|unauthori[sz]ed|forbidden", re.I)


class ProviderWatch:
    def __init__(self, net: Any, consecutive: int = 5, repeat_hours: float = 6.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.net, self.consecutive, self.repeat_s, self.clock = net, consecutive, repeat_hours * 3600, clock
        self.failures = 0
        self.alerted_at: float | None = None
        net.listeners.append(self.on_log)

    def on_log(self, e: dict[str, Any]) -> None:
        if e.get("event") != "episode_end":
            return
        if e.get("stopped") == "provider_error":
            self.failures += 1
            why = str(e.get("summary") or "")
            due = AUTH.search(why) is not None or self.failures >= self.consecutive
            now = self.clock()
            if due and (self.alerted_at is None or now - self.alerted_at >= self.repeat_s):
                self.alerted_at = now
                hint = (" API 키가 만료되었거나 잘못된 것 같습니다. 서버 .env의 키를 새로 넣고 "
                        "aichain-bridge를 재시작해 주세요." if AUTH.search(why) else "")
                self.net.manager_send("core", f"⚠️ LLM 호출이 실패하고 있습니다 (연속 {self.failures}회). "
                                              f"노드들이 생각하지 못하는 상태입니다.{hint}\n오류: {why[:300]}")
        elif e.get("llm_calls"):
            if self.alerted_at is not None:
                self.net.manager_send("core", f"✅ LLM 호출이 다시 됩니다 (그 전까지 실패 {self.failures}회).")
            self.failures, self.alerted_at = 0, None
