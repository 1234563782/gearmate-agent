import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import (
    ModelMessage,
    ModelRequest,
    ModelToolDefinition,
    ModelUsage,
)
from gearmate.tools.contracts import RentalPeriodInput

TEMPORAL_SIGNAL = re.compile(
    r"(?:今天|明天|后天|大后天|本周|这周|下周|周末|星期[一二三四五六日天]?|"
    r"周[一二三四五六日天]|月底|月初|早上|上午|中午|下午|晚上|凌晨|"
    r"\d{4}\s*[年/-]\s*\d{1,2}|\d{1,2}\s*月\s*\d{1,2}\s*[日号]?|"
    r"\d{1,2}\s*[点时]|[零〇一二两三四五六七八九十百\d]+\s*(?:天|小时|周|个?月)|"
    r"today|tomorrow|tonight|"
    r"next\s+(?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday))",
    re.IGNORECASE,
)

EXPLICIT_TIME_RANGE = re.compile(
    r"(?:[零〇一二两三四五六七八九十百\d]{1,3}\s*(?:点|时)"
    r"(?:\s*[零〇一二两三四五六七八九十百\d]{1,3}\s*分?)?.*?"
    r"(?:到|至|~|—|\uff0d)\s*.*?"
    r"[零〇一二两三四五六七八九十百\d]{1,3}\s*(?:点|时)"
    r"(?:\s*[零〇一二两三四五六七八九十百\d]{1,3}\s*分?)?|"
    r"\d{1,2}:\d{2}.*?(?:到|至|~|—|\uff0d)\s*.*?\d{1,2}:\d{2}|"
    r"\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2}.*?"
    r"(?:到|至|~|—|\uff0d).*?\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2})",
    re.IGNORECASE,
)

RESOLVER_TOOL_NAME = "set_rental_period"


@dataclass(frozen=True, slots=True)
class RentalPeriodResolution:
    rental_period: RentalPeriodInput | None
    clarification: str | None
    usage: ModelUsage


class InvalidRentalPeriod(ValueError):
    pass


class RentalPeriodPolicy:
    def __init__(self, max_advance_days: int) -> None:
        self._max_advance_days = max_advance_days

    @property
    def max_advance_days(self) -> int:
        return self._max_advance_days

    def validate(
        self,
        rental_period: RentalPeriodInput,
        *,
        now_utc: datetime | None = None,
    ) -> RentalPeriodInput:
        reference = now_utc or datetime.now(UTC)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        reference = reference.astimezone(UTC)
        start_at = rental_period.start_at.astimezone(UTC)
        if start_at <= reference:
            raise InvalidRentalPeriod("租赁开始时间必须晚于当前时间，请重新确认。")
        latest_start = reference + timedelta(days=self._max_advance_days)
        if start_at > latest_start:
            raise InvalidRentalPeriod(
                f"租赁开始时间不能超过未来 {self._max_advance_days} 天，请重新确认。"
            )
        return rental_period


def has_temporal_signal(message: str) -> bool:
    return TEMPORAL_SIGNAL.search(message) is not None


def has_explicit_time_range(message: str) -> bool:
    return EXPLICIT_TIME_RANGE.search(message) is not None


def resolver_system_prompt(
    timezone: str,
    now_utc: datetime,
    now_local: datetime,
    max_advance_days: int,
) -> str:
    return f"""你只负责从租赁对话中解析明确的开始时间和结束时间。
可信 UTC 时间: {now_utc.isoformat()}
用户时区: {timezone}
用户当地时间: {now_local.isoformat()}

规则:
1. 相对日期必须以上述用户当地时间为基准。
2. 只有开始日期、开始时间、结束日期、结束时间都能唯一确定时, 才调用 {RESOLVER_TOOL_NAME};
   一旦四项都明确, 必须调用工具, 不要再用文字重复确认。
3. 输出时间必须包含用户时区对应的 UTC offset。
4. "上午"、"下午"、"晚上"、"周末"等没有具体时分的表达不算完整。
5. 信息不完整或有多种解释时不要调用工具, 只返回一个简洁的中文确认问题。
6. 不要搜索商品、查询库存、计算价格或回答其他问题。
7. 租赁开始时间不得超过当前时间之后 {max_advance_days} 天。"""


class RentalPeriodResolver:
    def __init__(self, policy: RentalPeriodPolicy) -> None:
        self._policy = policy

    async def resolve(
        self,
        *,
        message: str,
        timezone: str,
        now_utc: datetime,
        now_local: datetime,
        history: tuple[ModelMessage, ...],
        model: ChatModelPort,
        max_output_tokens: int,
    ) -> RentalPeriodResolution:
        recent_history = [item for item in history if item.role in ("user", "assistant")][-6:]
        if (
            not recent_history
            or recent_history[-1].role != "user"
            or recent_history[-1].content != message
        ):
            recent_history.append(ModelMessage(role="user", content=message))
        request = ModelRequest(
            messages=(
                ModelMessage(
                    role="system",
                    content=resolver_system_prompt(
                        timezone,
                        now_utc,
                        now_local,
                        self._policy.max_advance_days,
                    ),
                ),
                *recent_history,
            ),
            tools=(
                ModelToolDefinition(
                    name=RESOLVER_TOOL_NAME,
                    description="保存用户已经明确确认的租赁开始和结束时间。",
                    parameters=RentalPeriodInput.model_json_schema(by_alias=True),
                ),
            ),
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            tool_choice=(
                RESOLVER_TOOL_NAME if has_explicit_time_range(message) else "auto"
            ),
            enable_thinking=False,
        )
        response = await model.complete(request)
        for call in response.tool_calls:
            if call.name != RESOLVER_TOOL_NAME:
                continue
            try:
                period = RentalPeriodInput.model_validate(call.arguments)
            except ValidationError:
                return RentalPeriodResolution(
                    rental_period=None,
                    clarification="请确认完整的开始日期时间和结束日期时间。",
                    usage=response.usage,
                )
            zone = ZoneInfo(timezone)
            if (
                period.start_at.utcoffset() != period.start_at.astimezone(zone).utcoffset()
                or period.end_at.utcoffset() != period.end_at.astimezone(zone).utcoffset()
            ):
                return RentalPeriodResolution(
                    rental_period=None,
                    clarification=f"请按 {timezone} 时区确认开始和结束时间。",
                    usage=response.usage,
                )
            try:
                self._policy.validate(period, now_utc=now_utc)
            except InvalidRentalPeriod as error:
                return RentalPeriodResolution(
                    rental_period=None,
                    clarification=str(error),
                    usage=response.usage,
                )
            return RentalPeriodResolution(
                rental_period=period,
                clarification=None,
                usage=response.usage,
            )
        clarification = response.text.strip() or ("请确认具体的开始日期时间和结束日期时间。")
        return RentalPeriodResolution(
            rental_period=None,
            clarification=clarification,
            usage=response.usage,
        )
