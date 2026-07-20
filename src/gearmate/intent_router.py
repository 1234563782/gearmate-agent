import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from gearmate.actions import AgentAction, PendingRentalAction
from gearmate.rental_period import has_temporal_signal

IntentPreRouteRule = Literal[
    "pure_social",
    "pending_confirmation",
    "pending_date_supplement",
]


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
_PENDING_CONFIRMATIONS = frozenset({"是", "对", "可以", "确认", "好的", "ok", "yes"})

_DATE_OR_TIME_TOKEN = (
    r"(?:"
    r"今天|明天|后天|大后天|本周|这周|下周|周末|月底|月初|"
    r"星期[一二三四五六日天]?|周[一二三四五六日天]|"
    r"\d{4}\s*(?:年|[./-])\s*\d{1,2}(?:\s*(?:月|[./-])\s*\d{1,2}\s*[日号]?)?|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]?|"
    r"\d{1,2}\s*[日号]|"
    r"(?:上午|下午|中午|晚上|夜里|凌晨)?\s*"
    r"[零〇一二两三四五六七八九十百\d]{1,4}\s*(?:点(?:半|\d{1,2}\s*分?)?|时|:\s*\d{1,2})|"
    r"[零〇一二两三四五六七八九十百\d]+\s*(?:天|周|个?月)|"
    r"today|tomorrow|tonight|next\s+"
    r"(?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"\d{1,4}(?:[./-]\d{1,2}){1,2}|\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
    r"\d+\s*(?:days?|weeks?|months?)"
    r")"
)
_TEMPORAL_CONNECTOR = (
    r"(?:从|自|到|至|起|止|开始|结束|归还|返回|租|租期|日期|时间|"
    r"改成|改到|改为|就是|定为|确认|为|和|以及|再|然后|"
    r"from|to|until|through|at|on|for|start|end|return|rent|confirm|and)"
)
_TEMPORAL_ONLY = re.compile(
    rf"^(?:{_DATE_OR_TIME_TOKEN}|{_TEMPORAL_CONNECTOR}|"
    rf"[\s,\uFF0C.\u3002!\uFF01?\uFF1F;\uFF1B:\uFF1A\u3001~\uFF5E\u2014\u2013-])+$",
    re.IGNORECASE,
)


def _normalize(message: str) -> str:
    return unicodedata.normalize("NFKC", message).strip().casefold()


def _compact(message: str) -> str:
    return _IGNORABLE_PUNCTUATION.sub("", _normalize(message))


def _is_temporal_only(message: str) -> bool:
    normalized = _normalize(message)
    return bool(
        normalized
        and has_temporal_signal(normalized)
        and _TEMPORAL_ONLY.fullmatch(normalized)
    )


class IntentPreRouter:
    def __init__(
        self,
        *,
        pure_social_enabled: bool = True,
        pending_confirmation_enabled: bool = True,
        pending_date_enabled: bool = True,
    ) -> None:
        self._pure_social_enabled = pure_social_enabled
        self._pending_confirmation_enabled = pending_confirmation_enabled
        self._pending_date_enabled = pending_date_enabled

    def resolve(
        self,
        message: str,
        *,
        pending_rental_action: PendingRentalAction | None,
    ) -> IntentPreRouteDecision | None:
        compact = _compact(message)
        if (
            self._pending_confirmation_enabled
            and pending_rental_action is not None
            and compact in _PENDING_CONFIRMATIONS
        ):
            return IntentPreRouteDecision(
                action=AgentAction(action="chat", continues_pending=True),
                rule="pending_confirmation",
            )
        if (
            self._pending_date_enabled
            and pending_rental_action is not None
            and _is_temporal_only(message)
        ):
            return IntentPreRouteDecision(
                action=AgentAction(action="chat", continues_pending=True),
                rule="pending_date_supplement",
            )
        if self._pure_social_enabled and compact in _PURE_SOCIAL:
            return IntentPreRouteDecision(
                action=AgentAction(action="chat"),
                rule="pure_social",
            )
        return None
