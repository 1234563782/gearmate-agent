import pytest
from pydantic import ValidationError

from gearmate.config import Settings
from gearmate.intent_router import IntentPreRouter


@pytest.mark.parametrize("message", ["你好", "您好，谢谢", "HELLO!", "好的"])
def test_pure_social_requires_a_whole_message_match(message: str) -> None:
    decision = IntentPreRouter().resolve(message)

    assert decision is not None
    assert decision.rule == "pure_social"
    assert decision.action.action == "chat"


@pytest.mark.parametrize(
    "message",
    [
        "你好，帮我查订单",
        "谢谢，第一台有货吗",
        "hi, need a camera",
        "2026-07-20 到 2026-07-22",
    ],
)
def test_business_or_date_content_falls_back_to_the_llm(message: str) -> None:
    assert IntentPreRouter().resolve(message) is None


def test_pure_social_rule_can_be_disabled() -> None:
    assert IntentPreRouter(pure_social_enabled=False).resolve("你好") is None


def test_pre_router_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, intent_pre_router_mode="invalid")  # type: ignore[arg-type]
