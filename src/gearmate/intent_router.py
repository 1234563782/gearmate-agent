import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from gearmate.actions import AgentAction

IntentPreRouteRule = Literal["pure_social"]


@dataclass(frozen=True, slots=True)
class IntentPreRouteDecision:
    action: AgentAction
    rule: IntentPreRouteRule


_IGNORABLE_PUNCTUATION = re.compile(
    r"[\s,\uFF0C.\u3002!\uFF01?\uFF1F;\uFF1B:\uFF1A\u3001'\""
    r"\u201c\u201d\u2018\u2019]+"
)
_PURE_SOCIAL = frozenset(
    {
        "你好",
        "您好",
        "hello",
        "hi",
        "谢谢",
        "感谢",
        "多谢",
        "不客气",
        "好的",
        "你好谢谢",
        "您好谢谢",
    }
)


def _normalize(message: str) -> str:
    return unicodedata.normalize("NFKC", message).strip().casefold()


def _compact(message: str) -> str:
    return _IGNORABLE_PUNCTUATION.sub("", _normalize(message))


class IntentPreRouter:
    def __init__(self, *, pure_social_enabled: bool = True) -> None:
        self._pure_social_enabled = pure_social_enabled

    def resolve(self, message: str) -> IntentPreRouteDecision | None:
        if self._pure_social_enabled and _compact(message) in _PURE_SOCIAL:
            return IntentPreRouteDecision(
                action=AgentAction(action="chat"),
                rule="pure_social",
            )
        return None
