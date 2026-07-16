from datetime import UTC, datetime

from gearmate.llm.types import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)
from gearmate.rental_period import (
    InvalidRentalPeriod,
    RentalPeriodPolicy,
    RentalPeriodResolver,
    has_explicit_time_range,
    has_temporal_signal,
)
from gearmate.tools.contracts import RentalPeriodInput


class FakeModel:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.response

    async def close(self) -> None:
        return None


def model_response(
    *,
    text: str = "",
    arguments: dict[str, object] | None = None,
) -> ModelResponse:
    calls = (
        (
            ModelToolCall(
                id="call-1",
                name="set_rental_period",
                arguments=arguments,
            ),
        )
        if arguments is not None
        else ()
    )
    return ModelResponse(
        text=text,
        finish_reason="tool_calls" if calls else "stop",
        usage=ModelUsage(input_tokens=80, output_tokens=12),
        tool_calls=calls,
    )


async def test_resolver_accepts_complete_period_in_user_timezone() -> None:
    model = FakeModel(
        model_response(
            arguments={
                "startAt": "2026-07-17T09:00:00+08:00",
                "endAt": "2026-07-19T18:00:00+08:00",
            }
        )
    )
    resolver = RentalPeriodResolver(RentalPeriodPolicy(90))
    now_utc = datetime(2026, 7, 15, 2, 30, tzinfo=UTC)

    result = await resolver.resolve(
        message="本周五早上 9 点到周日 18 点",
        timezone="Asia/Shanghai",
        now_utc=now_utc,
        now_local=datetime.fromisoformat("2026-07-15T10:30:00+08:00"),
        history=(),
        model=model,
        max_output_tokens=256,
    )

    assert result.rental_period == RentalPeriodInput(
        start_at=datetime.fromisoformat("2026-07-17T09:00:00+08:00"),
        end_at=datetime.fromisoformat("2026-07-19T18:00:00+08:00"),
    )
    assert result.clarification is None
    assert "用户当地时间: 2026-07-15T10:30:00+08:00" in (model.requests[0].messages[0].content)
    assert model.requests[0].enable_thinking is False
    assert model.requests[0].tool_choice == "set_rental_period"


async def test_resolver_returns_clarification_for_ambiguous_period() -> None:
    model = FakeModel(model_response(text="明天下午具体几点开始和结束？"))
    resolver = RentalPeriodResolver(RentalPeriodPolicy(90))

    result = await resolver.resolve(
        message="明天下午租一台相机",
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 15, 2, 30, tzinfo=UTC),
        now_local=datetime.fromisoformat("2026-07-15T10:30:00+08:00"),
        history=(),
        model=model,
        max_output_tokens=256,
    )

    assert result.rental_period is None
    assert result.clarification == "明天下午具体几点开始和结束？"


async def test_resolver_rejects_wrong_timezone_offset() -> None:
    model = FakeModel(
        model_response(
            arguments={
                "startAt": "2026-07-17T09:00:00+00:00",
                "endAt": "2026-07-19T18:00:00+00:00",
            }
        )
    )
    resolver = RentalPeriodResolver(RentalPeriodPolicy(90))

    result = await resolver.resolve(
        message="7 月 17 日 9 点到 19 日 18 点",
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 15, 2, 30, tzinfo=UTC),
        now_local=datetime.fromisoformat("2026-07-15T10:30:00+08:00"),
        history=(),
        model=model,
        max_output_tokens=256,
    )

    assert result.rental_period is None
    assert result.clarification == "请按 Asia/Shanghai 时区确认开始和结束时间。"


def test_temporal_signal_detection() -> None:
    assert has_temporal_signal("明天下午租相机")
    assert has_temporal_signal("7 月 20 日 9 点开始")
    assert has_temporal_signal("我想租三天")
    assert has_temporal_signal("大概用两小时")
    assert not has_temporal_signal("有哪些佳能相机？")


def test_explicit_time_range_detection_requires_two_clock_times() -> None:
    assert has_explicit_time_range("7 月 20 日上午 10 点到 7 月 21 日下午 6 点")
    assert has_explicit_time_range("从今天下午六点到明天下午六点")
    assert has_explicit_time_range("2026-07-20 10:00 至 2026-07-21 18:00")
    assert not has_explicit_time_range("7 月 20 日下午租相机")


def test_policy_accepts_90_day_boundary_and_rejects_after_it() -> None:
    policy = RentalPeriodPolicy(90)
    now = datetime(2026, 7, 15, 2, 30, tzinfo=UTC)
    boundary = RentalPeriodInput(
        start_at=datetime(2026, 10, 13, 2, 30, tzinfo=UTC),
        end_at=datetime(2026, 10, 14, 2, 30, tzinfo=UTC),
    )

    assert policy.validate(boundary, now_utc=now) == boundary

    outside = boundary.model_copy(update={"start_at": datetime(2026, 10, 13, 2, 31, tzinfo=UTC)})
    try:
        policy.validate(outside, now_utc=now)
    except InvalidRentalPeriod as error:
        assert "未来 90 天" in str(error)
    else:
        raise AssertionError("a rental starting after the boundary must fail")


async def test_resolver_rejects_period_beyond_advance_window() -> None:
    model = FakeModel(
        model_response(
            arguments={
                "startAt": "2026-10-14T09:00:00+08:00",
                "endAt": "2026-10-15T18:00:00+08:00",
            }
        )
    )
    resolver = RentalPeriodResolver(RentalPeriodPolicy(90))

    result = await resolver.resolve(
        message="10 月 14 日 9 点到 15 日 18 点",
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 15, 2, 30, tzinfo=UTC),
        now_local=datetime.fromisoformat("2026-07-15T10:30:00+08:00"),
        history=(),
        model=model,
        max_output_tokens=256,
    )

    assert result.rental_period is None
    assert result.clarification is not None
    assert "未来 90 天" in result.clarification
