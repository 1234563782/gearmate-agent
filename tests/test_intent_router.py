import pytest
from pydantic import ValidationError

from gearmate.actions import PendingRentalAction
from gearmate.config import Settings
from gearmate.intent_router import IntentPreRouter

PENDING = PendingRentalAction(
    action="availability",
    product_id="01J00000000000000000000101",
)


@pytest.mark.parametrize("message", ["你好", "您好，谢谢", "HELLO!", "好的"])
def test_pure_social_requires_a_whole_message_match(message: str) -> None:
    decision = IntentPreRouter().resolve(message, pending_rental_action=None)

    assert decision is not None
    assert decision.rule == "pure_social"
    assert decision.action.action == "chat"
    assert decision.action.continues_pending is False


@pytest.mark.parametrize(
    "message",
    [
        "你好，帮我查订单",
        "谢谢，第一台有货吗",
        "hi, need a camera",
    ],
)
def test_business_content_falls_back_to_the_llm(message: str) -> None:
    assert IntentPreRouter().resolve(message, pending_rental_action=None) is None


def test_pending_confirmation_has_priority_over_pure_social() -> None:
    decision = IntentPreRouter().resolve("好的", pending_rental_action=PENDING)

    assert decision is not None
    assert decision.rule == "pending_confirmation"
    assert decision.action.continues_pending is True


@pytest.mark.parametrize(
    "message",
    [
        "2026-07-20 到 2026-07-22",
        "从今天下午六点到明天下午六点",
        "后天开始租三天",
    ],
)
def test_date_only_message_continues_a_pending_rental_action(message: str) -> None:
    decision = IntentPreRouter().resolve(message, pending_rental_action=PENDING)

    assert decision is not None
    assert decision.rule == "pending_date_supplement"
    assert decision.action.continues_pending is True


def test_date_only_message_without_pending_action_falls_back() -> None:
    assert (
        IntentPreRouter().resolve(
            "2026-07-20 到 2026-07-22",
            pending_rental_action=None,
        )
        is None
    )


@pytest.mark.parametrize(
    "message",
    [
        "第一台 7 月 20 到 22 能租吗",
        "换手机，7 月 20 到 22",
        "刚才第一个 7 月 20 到 22 能租吗",
    ],
)
def test_mixed_temporal_business_message_falls_back_to_the_llm(message: str) -> None:
    assert IntentPreRouter().resolve(message, pending_rental_action=PENDING) is None


def test_individual_rules_can_be_disabled() -> None:
    router = IntentPreRouter(
        pure_social_enabled=False,
        pending_confirmation_enabled=False,
        pending_date_enabled=False,
    )

    assert router.resolve("你好", pending_rental_action=None) is None
    assert router.resolve("好的", pending_rental_action=PENDING) is None
    assert router.resolve("2026-07-20 到 2026-07-22", pending_rental_action=PENDING) is None


def test_pre_router_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, intent_pre_router_mode="invalid")  # type: ignore[arg-type]
