import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from gearmate.llm.port import ChatModelPort
from gearmate.llm.types import ModelMessage, ModelRequest, ModelToolDefinition, ModelUsage
from gearmate.tools.contracts import RentalPeriodInput

BUSINESS_ZONE = ZoneInfo("Asia/Shanghai")
TEMPORAL_SIGNAL = re.compile(
    r"(?:今天|明天|后天|大后天|本周|这周|下周|周末|星期[一二三四五六日天]?|"
    r"周[一二三四五六日天]|月底|月初|\d{4}\s*[年/-]\s*\d{1,2}|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]?|[零〇一二两三四五六七八九十百\d]+\s*(?:天|周|个?月)|"
    r"today|tomorrow|next\s+(?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday))",
    re.IGNORECASE,
)
EXPLICIT_DATE_RANGE = re.compile(
    r"(?:\d{4}\s*[年/-]\s*\d{1,2}\s*[月/-]\s*\d{1,2}\s*[日号]?|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]?|今天|明天|后天|大后天).*?"
    r"(?:到|至|~|—|-).*?"
    r"(?:\d{4}\s*[年/-]\s*\d{1,2}\s*[月/-]\s*\d{1,2}\s*[日号]?|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]?|今天|明天|后天|大后天)",
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
    def __init__(self, max_advance_days: int, max_rental_days: int = 30) -> None:
        self._max_advance_days = max_advance_days
        self._max_rental_days = max_rental_days

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
        today = reference.astimezone(BUSINESS_ZONE).date()
        earliest_start = today + timedelta(days=2)
        if rental_period.start_date < earliest_start:
            raise InvalidRentalPeriod("租赁开始日期最早为后天（按北京时间），请重新确认。")
        if rental_period.start_date > today + timedelta(days=self._max_advance_days):
            raise InvalidRentalPeriod(
                f"租赁开始日期不能超过未来 {self._max_advance_days} 天，请重新确认。"
            )
        if rental_period.billing_days > self._max_rental_days:
            raise InvalidRentalPeriod(
                f"租期最多为 {self._max_rental_days} 个自然日（归还日期包含当天），请重新确认。"
            )
        return rental_period


def has_temporal_signal(message: str) -> bool:
    return TEMPORAL_SIGNAL.search(message) is not None


def has_explicit_date_range(message: str) -> bool:
    return EXPLICIT_DATE_RANGE.search(message) is not None


def has_explicit_time_range(message: str) -> bool:
    return has_explicit_date_range(message)


def resolver_system_prompt(
    timezone: str,
    now_utc: datetime,
    now_local: datetime,
    max_advance_days: int,
) -> str:
    business_now = now_utc.astimezone(BUSINESS_ZONE)
    return f"""你只负责从租赁对话中解析明确的开始日期和归还日期。
可信 UTC 时间: {now_utc.isoformat()}
上海租赁日历当前日期: {business_now.date().isoformat()}
用户时区: {timezone}
用户当地时间: {now_local.isoformat()}

规则:
1. 租赁只按自然日计算，业务日历固定为 Asia/Shanghai；不要输出时分秒或 UTC offset。
2. 输出必须是 YYYY-MM-DD 格式的 startDate 和 endDate；归还日期包含在租期内。
3. 相对日期必须以上述上海租赁日历当前日期为基准。
4. 只有开始日期和归还日期都能唯一确定时，才调用 {RESOLVER_TOOL_NAME}；
   一旦两项都明确，必须调用工具，不要再用文字重复确认。
5. 开始日期最早为上海日期的后天，且不得超过未来 {max_advance_days} 天；单日租赁有效。
6. 信息不完整或有多种解释时不要调用工具，只返回一个简洁的中文确认问题。
7. 不要搜索商品、查询库存、计算价格或回答其他问题。"""


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
                        timezone, now_utc, now_local, self._policy.max_advance_days
                    ),
                ),
                *recent_history,
            ),
            tools=(
                ModelToolDefinition(
                    name=RESOLVER_TOOL_NAME,
                    description="保存用户已经明确确认的租赁开始日期和归还日期。",
                    parameters=RentalPeriodInput.model_json_schema(by_alias=True),
                ),
            ),
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            tool_choice=RESOLVER_TOOL_NAME if has_explicit_date_range(message) else "auto",
            enable_thinking=False,
            workload="action",
        )
        response = await model.complete(request)
        for call in response.tool_calls:
            if call.name != RESOLVER_TOOL_NAME:
                continue
            try:
                period = RentalPeriodInput.model_validate(call.arguments)
                self._policy.validate(period, now_utc=now_utc)
            except (ValidationError, InvalidRentalPeriod):
                return RentalPeriodResolution(
                    rental_period=None,
                    clarification="请确认完整的开始日期和归还日期；最早可从后天开始租，归还日期包含当天。",
                    usage=response.usage,
                )
            return RentalPeriodResolution(
                rental_period=period, clarification=None, usage=response.usage
            )
        clarification = response.text.strip() or "请确认具体的开始日期和归还日期。"
        return RentalPeriodResolution(
            rental_period=None, clarification=clarification, usage=response.usage
        )
